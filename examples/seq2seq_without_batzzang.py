# used codes from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math
import random
import argparse
import itertools
from load_and_trim_data import EOS_token, PAD_token, SOS_token, MAX_LENGTH, loadPrepareData, trimRareWords
from create_formatted_data_file import create_formatted_data_file
from transformers import BertModel, BertTokenizer


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc, tokenizer=None):
    if tokenizer:
        indexes_batch = tokenizer(l, return_tensors='pt', padding=True).to("cuda")
        lengths = torch.tensor([len(input_ids) for input_ids in indexes_batch.data['input_ids']])
        return indexes_batch, lengths
    else:
        indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = zeroPadding(indexes_batch)
        padVar = torch.tensor(padList, dtype=torch.long, device=device)
        return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask).to(device)
    padVar = torch.tensor(padList, dtype=torch.long, device=device)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, model):
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc, tokenizer=tokenizer)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class GRUSeq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_num, hidden_size, layer_num, dropout):
        super(GRUSeq2SeqWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_num, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder_gru = nn.GRU(hidden_size, hidden_size, layer_num,
                                  dropout=(0 if layer_num == 1 else dropout), bidirectional=True)
        self.decoder_gru = nn.GRU(hidden_size, hidden_size, layer_num,
                                  dropout=(0 if layer_num == 1 else dropout), bidirectional=False)

        self.attn = Attn(hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_num)
        self.hidden_size = hidden_size
        self.layer_num = layer_num

    def _embed_encoder_input(self, input_variable):
        embedded = self.embedding(input_variable)
        return self.dropout(embedded)

    def _forward_encoder(self, embedded, input_lengths):
        # Lengths for rnn packing should always be on the cpu
        input_lengths = input_lengths.to("cpu")
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        encoder_outputs, encoder_hidden = self.encoder_gru(packed, None)
        # Unpack padding
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs)
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        return encoder_outputs, encoder_hidden

    def _embed_decoder_input(self, decoder_input):
        embedded = self.embedding(decoder_input)
        return self.dropout(embedded)

    def _forward_decoder(self, embedded, decoder_hidden):
        # Forward through unidirectional GRU
        gru_output, hidden = self.decoder_gru(embedded, decoder_hidden)
        return gru_output, hidden

    def _compute_attention(self, gru_output, encoder_outputs):
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(gru_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        context = context.squeeze(1)
        concat_input = torch.cat((gru_output.squeeze(0), context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        return concat_output

    def _compute_output_prob(self, attention_output):
        output = self.out(attention_output)
        return F.softmax(output, dim=1)

    def forward_no_teacher_forcing(self, input_variable, input_lengths, target_variable, mask, max_target_len):
        bsz = len(input_lengths)
        decoder_input = torch.tensor([[SOS_token for _ in range(bsz)]], dtype=torch.long, device=device)

        # Embed encoder input
        embedded_encoder_input = self._embed_encoder_input(input_variable)

        # forward encoder
        encoder_outputs, encoder_hidden = self._forward_encoder(embedded_encoder_input, input_lengths)
        encoder_hidden = encoder_hidden.view(self.layer_num, 2, bsz, self.hidden_size)
        decoder_hidden = encoder_hidden[:, 0] + encoder_hidden[:, 1]

        loss = 0
        for t in range(max_target_len):
            # embed decoder input
            embedded = self._embed_decoder_input(decoder_input)

            # forward decoder
            gru_output, decoder_hidden = self._forward_decoder(embedded, decoder_hidden)

            # compute attention
            attention_output = self._compute_attention(gru_output, encoder_outputs)

            # compute output prob
            output_prob = self._compute_output_prob(attention_output)

            # No Teacher forcing
            _, topi = output_prob.topk(1)

            decoder_input = torch.tensor([[topi[i][0] for i in range(bsz)]], dtype=torch.long, device=device)

            # calc loss
            mask_loss, nTotal = maskNLLLoss(output_prob, target_variable[t], mask[t])
            loss += mask_loss
        return loss

    def forward_teacher_forcing(self, input_variable, input_lengths, target_variable, mask, max_target_len):
        assert False, "TODO"


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_num, hidden_size, layer_num, dropout):
        super(TransformerSeq2Seq, self).__init__()
        nhead = 8
        dim_feedforward = 1024
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=layer_num)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=layer_num)
        self._init_positional_embedding("encoder_pe", hidden_size, dropout)
        self._init_positional_embedding("decoder_pe", hidden_size, dropout)

        self.embedding = nn.Embedding(vocab_num, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_num)
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout)

    def _init_positional_embedding(self, buffer_name, d_model, dropout=0.1, max_len=500,):
        self.pos_dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(buffer_name, pe)

    def _encoder_pos_encode(self, x):
        x = x + self.encoder_pe[: x.size(0), :]
        return self.pos_dropout(x)

    def _decoder_pos_encode(self, x):
        x = x + self.decoder_pe[: x.size(0), :]
        return self.pos_dropout(x)

    def _embed_encoder_input(self, encoder_input):
        embedded = self.embedding(encoder_input)
        return self.dropout(embedded)

    def _forward_encoder(self, embedded_input, input_mask):
        embedded_input = self._encoder_pos_encode(embedded_input)
        encoder_output = self.encoder_transformer(embedded_input, src_key_padding_mask=input_mask)
        return encoder_output

    def _embed_decoder_input(self, decoder_input):
        embedded = self.embedding(decoder_input)
        return self.dropout(embedded)

    def _forward_decoder(self, embedded_input, encoder_output, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        embedded_input = self._decoder_pos_encode(embedded_input)
        decoder_output = self.decoder_transformer(embedded_input, encoder_output, tgt_mask=tgt_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask)
        return decoder_output

    def _compute_output_prob(self, decoder_output):
        output = self.out(decoder_output)
        return F.softmax(output, dim=-1)

    def _calc_loss(self, output_prob, tgt_variable, tgt_padding_mask, input_lengths):
        flatten_decoder_output = output_prob.reshape(-1, output_prob.shape[-1])
        target_indices = (tgt_padding_mask.reshape(-1) != 0).nonzero()
        target_decoder_output = flatten_decoder_output[target_indices].squeeze(1)
        target_tgt_variable = tgt_variable.reshape(-1)[target_indices].squeeze(1)
        avg_len = sum(input_lengths) / len(input_lengths)
        loss = self.loss(target_decoder_output, target_tgt_variable) * avg_len
        return loss

    def forward_no_teacher_forcing(self, input_variable, input_lengths, target_variable, tgt_padding_mask, max_target_len):
        # Create masking and inputs
        bsz = len(input_lengths)
        input_padding_mask = (torch.arange(max(input_lengths))[None, :] > input_lengths[:, None]).cuda()
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.tensor([[SOS_token for _ in range(bsz)]], dtype=torch.long, device=device)

        ## Begin logic
        # Embed encoder input
        embedded_encoder_input = self._embed_encoder_input(input_variable)

        # Forward encoder
        encoder_outputs = self._forward_encoder(embedded_encoder_input, input_padding_mask)

        loss = 0.
        for t in range(max_target_len):
            # Embed_decoder_input
            embedded_decoder_input = self._embed_decoder_input(decoder_input)

            # Forward decoder
            decoder_output = self._forward_decoder(embedded_decoder_input, encoder_outputs,
                                                  memory_key_padding_mask=input_padding_mask).squeeze(0)

            # Compute_output_prob
            output_prob = self._compute_output_prob(decoder_output)
            _, topi = output_prob.topk(1)

            # Calculate loss
            mask_loss, nTotal = maskNLLLoss(output_prob, target_variable[t], tgt_padding_mask[t])
            loss += mask_loss

            # Next decoder input
            decoder_input = torch.tensor([[topi[i][0] for i in range(bsz)]], dtype=torch.long, device=device)
        return loss

    def forward_teacher_forcing(self, input_variable, input_lengths, tgt_variable, tgt_padding_mask, max_tgt_length):
        # Create masking and inputs
        bsz = len(input_lengths)
        input_padding_mask = (torch.arange(max(input_lengths))[None, :] > input_lengths[:, None]).cuda()
        # Create initial decoder input (start with SOS tokens for each sentence)
        init_input = torch.tensor([[SOS_token for _ in range(bsz)]], dtype=torch.long, device=device)
        decoder_input = torch.cat((init_input, tgt_variable[:-1]), dim=0)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(None, list(tgt_variable.size())[0]).cuda()

        ## Begin logic
        # Embed encoder input
        embedded_encoder_input = self._embed_encoder_input(input_variable)

        # Forward encoder
        encoder_outputs = self._forward_encoder(embedded_encoder_input, input_padding_mask)

        # Embed decoder input
        embedded_decoder_input = self._embed_decoder_input(decoder_input)

        # Forward decoder
        decoder_output = self._forward_decoder(embedded_decoder_input, encoder_outputs, tgt_mask,
                                              input_padding_mask, ~tgt_padding_mask.transpose(0,1))
        # Compute output prob
        output_prob = self._compute_output_prob(decoder_output)

        # Calculate loss
        loss = self._calc_loss(output_prob, tgt_variable, tgt_padding_mask, input_lengths)

        return loss


class BertSeq2Seq(nn.Module):
    def __init__(self, vocab_num, hidden_size, layer_num, dropout):
        super(BertSeq2Seq, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=1024, dropout=dropout)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=layer_num)
        self.embedding = nn.Embedding(vocab_num, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_num)
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout)

    def decode(self, decoder_input, encoder_output, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        embedded = self.embedding(decoder_input)
        embedded = self.dropout(embedded)
        decoder_output = self.decoder_transformer(embedded, encoder_output, tgt_mask=tgt_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.out(decoder_output)
        output = F.softmax(output, dim=-1)
        return output

    def _forward_encoder(self, input_variable):
        bert_output = self.bert(**input_variable)
        return bert_output['last_hidden_state'].transpose(0, 1)

    def _embed_decoder_input(self, decoder_input):
        embedded = self.embedding(decoder_input)
        return self.dropout(embedded)

    def _forward_decoder(self, embedded, encoder_output, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        decoder_output = self.decoder_transformer(embedded, encoder_output, tgt_mask=tgt_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask)
        return decoder_output

    def _compute_output_prob(self, decoder_output):
        output = self.out(decoder_output)
        return F.softmax(output, dim=-1)

    def _calc_loss(self, output_prob, tgt_variable, tgt_padding_mask, input_lengths):
        flatten_decoder_output = output_prob.reshape(-1, output_prob.shape[-1])
        target_indices = (tgt_padding_mask.reshape(-1) != 0).nonzero()
        target_decoder_output = flatten_decoder_output[target_indices].squeeze(1)
        target_tgt_variable = tgt_variable.reshape(-1)[target_indices].squeeze(1)
        avg_len = sum(input_lengths) / len(input_lengths)
        loss = self.loss(target_decoder_output, target_tgt_variable) * avg_len
        return loss


    def forward_no_teacher_forcing(self, input_variable, input_lengths, target_variable, tgt_padding_mask, max_target_len):
        # Create masking and inputs
        bsz = len(input_lengths)
        input_padding_mask = (torch.arange(max(input_lengths))[None, :] > input_lengths[:, None]).cuda()
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.tensor([[SOS_token for _ in range(bsz)]], dtype=torch.long, device=device)

        ## Begin logic
        # Forward encoder
        encoder_outputs = self._forward_encoder(input_variable)

        loss = 0.
        for t in range(max_target_len):
            # Embed_decoder_input
            embedded_decoder_input = self._embed_decoder_input(decoder_input)

            # Forward decoder
            decoder_output = self._forward_decoder(embedded_decoder_input, encoder_outputs,
                                                   memory_key_padding_mask=input_padding_mask).squeeze(0)

            # Compute_output_prob
            output_prob = self._compute_output_prob(decoder_output)
            _, topi = output_prob.topk(1)

            # Calculate loss
            mask_loss, nTotal = maskNLLLoss(output_prob, target_variable[t], tgt_padding_mask[t])
            loss += mask_loss

            # Next decoder input
            decoder_input = torch.tensor([[topi[i][0] for i in range(bsz)]], dtype=torch.long, device=device)

        return loss

    def forward_teacher_forcing(self, input_variable, input_lengths, tgt_variable, tgt_padding_mask, max_tgt_length):
        # Create masking and input
        bsz = len(input_lengths)
        input_padding_mask = (torch.arange(max(input_lengths))[None, :] > input_lengths[:, None]).cuda()
        # Create initial decoder input (start with SOS tokens for each sentence)
        init_input = torch.tensor([[SOS_token for _ in range(bsz)]], dtype=torch.long, device=device)
        decoder_input = torch.cat((init_input, tgt_variable[:-1]), dim=0)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(None, list(tgt_variable.size())[0]).cuda()

        ## Begin logic
        # Forward encoder
        encoder_outputs = self._forward_encoder(input_variable)

        # Embed_decoder_input
        embedded_decoder_input = self._embed_decoder_input(decoder_input)

        # Forward decoder
        decoder_output = self._forward_decoder(embedded_decoder_input, encoder_outputs, tgt_mask,
                                               input_padding_mask, ~tgt_padding_mask.transpose(0,1))

        # Compute output prob
        output_prob = self._compute_output_prob(decoder_output)

        # Calculate loss
        loss = self._calc_loss(output_prob, tgt_variable, tgt_padding_mask, input_lengths)

        return loss


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, input_lengths, target_variable, mask, max_target_len, model, optimizer, clip, no_teacher_forcing):

    # Zero gradients
    optimizer.zero_grad()

    if no_teacher_forcing:
        loss = model.forward_no_teacher_forcing(input_variable, input_lengths, target_variable, mask, max_target_len)
    else:
        loss = model.forward_teacher_forcing(input_variable, input_lengths, target_variable, mask, max_target_len)

    # Perform backpropatation
    t1 = time.time()
    loss.backward()
    t2 = time.time()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    optimizer.step()

    return float(loss) / max_target_len, t2-t1


def trainIters(voc, pairs, model, optimizer, n_iteration, batch_size, print_every, clip, no_teacher_forcing):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], model)
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    start_time = time.time()
    total_backward_time = 0.0

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        avg_loss, backward_time = train(input_variable, lengths, target_variable, mask, max_target_len, model, optimizer, clip, no_teacher_forcing)
        print_loss += avg_loss
        total_backward_time += backward_time

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; time: {:.1f}, Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, time.time() - start_time, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
    print(f"Backward time:{total_backward_time}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gru', 'transformer', 'bert'])
    parser.add_argument('--no_teacher_forcing', action='store_true')
    args = parser.parse_args()

    create_formatted_data_file()
    voc, pairs = loadPrepareData()

    # Configurations
    hidden_size = 768 if args.model == 'bert' else 512
    layer_num = 2
    dropout = 0.1
    batch_size = 64
    clip = 50.0
    learning_rate = 0.0001
    n_iteration = 4000
    print_every = 1

    print('Building encoder and decoder ...')
    if args.model == 'gru':
        model = GRUSeq2SeqWithAttention(voc.num_words, hidden_size, layer_num, dropout)
    elif args.model == 'transformer':
        model = TransformerSeq2Seq(voc.num_words, hidden_size, layer_num, dropout)
    else:
        assert args.model == 'bert'
        model = BertSeq2Seq(voc.num_words, hidden_size, layer_num, dropout)

    # Initialize word embeddings
    model = model.to(device)
    model.train()
    print('Models built and ready to go!')

    # Initialize optimizers
    print('Building optimizers ...')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # If you have cuda, configure cuda to call
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    trainIters(voc, pairs, model, optimizer, n_iteration, batch_size, print_every, clip, args.no_teacher_forcing)


if __name__ == "__main__":
    main()
