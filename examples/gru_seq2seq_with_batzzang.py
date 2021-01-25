# used codes from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import itertools
import time
from batzzang.lazy_modules import LazyGRU, LazyEmbedding, LazyLinear, LazyGRUCell, LazyAttention
from batzzang.monad import ForEachState, While, If, Do
from batzzang.tensor_promise import TensorPromise
from load_and_trim_data import EOS_token, PAD_token, SOS_token, MAX_LENGTH, loadPrepareData, trimRareWords
from create_formatted_data_file import create_formatted_data_file


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# Returns padded input sequence tensor and lengths
def makeVar(l, voc):
    indexes_batch = [torch.LongTensor(indexesFromSentence(voc, sentence)).to(device) for sentence in l]
    return indexes_batch


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp = makeVar(input_batch, voc)
    output = makeVar(output_batch, voc)
    return inp, output


class State:
    def __init__(self, input_seq, target_seq):
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.decoder_input_token = torch.LongTensor([SOS_token]).to(device)
        self.decoding_cnt = 0
        self.loss = 0


class StateTeacherForcing:
    def __init__(self, input_seq, target_seq):
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.decoder_input = torch.cat((torch.LongTensor([SOS_token]).to(device), target_seq[:-1]), dim=0)

        self.loss = 0


class GRUSeq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_num, hidden_size, encoder_layers, decoder_layers, dropout):
        super(GRUSeq2SeqWithAttention, self).__init__()
        self.embedding = LazyEmbedding(vocab_num, hidden_size, dropout)
        self.encoder_gru = LazyGRU(hidden_size, hidden_size, encoder_layers, bidirection=True)
        self.decoder_gru = LazyGRU(hidden_size, hidden_size, decoder_layers, bidirection=False)
        self.att = LazyAttention(hidden_size, 'dot')
        self.out = LazyLinear(hidden_size, vocab_num)
        self.hidden_size = hidden_size
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_seq_batch, target_seq_batch):
        states = [
            State(input_seq, target_seq)
            for input_seq, target_seq in zip(input_seq_batch, target_seq_batch)
        ]

        def embed(state: State, _):
            return self.embedding.forward_later(state.input_seq)

        def encode(state: State, previous_output):
            embedded_tensor = previous_output.result
            return self.encoder_gru.forward_later(embedded_tensor)

        def is_not_done(state: State):
            return state.decoding_cnt < len(state.target_seq)

        def embed_decoder_input(state: State, previous_output):
            if state.decoding_cnt == 0:
                encoder_promise = previous_output
                encoder_output, encoder_hidden = encoder_promise.result
                encoder_output = encoder_output[:, :self.hidden_size] + encoder_output[:, self.hidden_size:]
                decoder_hidden = encoder_hidden[-1, 0] + encoder_hidden[-1, 1]
                decoder_input = state.decoder_input_token
            else:
                encoder_output, decoder_hidden = previous_output
                decoder_input = state.decoder_input_token

            decoder_input_embedded_promise = self.embedding.forward_later(decoder_input)
            return encoder_output, decoder_hidden, decoder_input_embedded_promise

        def decode_gru(state: State, previous_output):
            encoder_output, decoder_hidden, decoder_input_embedded_promise = previous_output
            decoder_input_embedded = decoder_input_embedded_promise.result
            return encoder_output, self.decoder_gru.forward_later(decoder_input_embedded, decoder_hidden)

        def attention(state: State, previous_output):
            encoder_output, decoder_hidden_promise = previous_output
            decoder_output, decoder_hidden = decoder_hidden_promise.result
            return encoder_output, decoder_hidden, self.att.forward_later(decoder_output, encoder_output)

        def out(state: State, previous_output):
            encoder_output, decoder_hidden, attention_promise = previous_output
            attention_output, _ = attention_promise.result
            output_promise = self.out.forward_later(attention_output)
            return encoder_output, decoder_hidden, output_promise

        def calc_loss(state: State, previous_output):
            encoder_output, decoder_hidden, output_promise = previous_output
            output = F.softmax(output_promise.result.squeeze(0), dim=-1)
            target_idx = state.target_seq[state.decoding_cnt]
            state.loss += -torch.log(output[target_idx])

            state.decoding_cnt += 1
            state.decoder_input_token = torch.argmax(output).unsqueeze(0)
            return encoder_output, decoder_hidden

        result_states = ForEachState(states).apply(
            Do(embed)
                .Then(encode)
        ).apply(
            While.Any(is_not_done,
                If(is_not_done)
                    .Then(embed_decoder_input)
                    .Then(decode_gru)
                    .Then(attention)
                    .Then(out)
                    .Then(calc_loss)
            )
        ).states

        return sum([state.loss / state.decoding_cnt for state in result_states]) / len(states)

    def forward_teacher_forcing(self, input_seq_batch, target_seq_batch):
        states = [
            StateTeacherForcing(input_seq, target_seq)
            for input_seq, target_seq in zip(input_seq_batch, target_seq_batch)
        ]

        def embed(state: StateTeacherForcing, _):
            return self.embedding.forward_later(state.input_seq)

        def encode(state: StateTeacherForcing, previous_output):
            embedded_tensor = previous_output.result
            return self.encoder_gru.forward_later(embedded_tensor)

        def embed_decoder_input(state: StateTeacherForcing, previous_output):
            encoder_promise = previous_output
            encoder_output, encoder_hidden = encoder_promise.result
            encoder_output = encoder_output[:, :self.hidden_size] + encoder_output[:, self.hidden_size:]
            decoder_hidden = encoder_hidden[-1, 0] + encoder_hidden[-1, 1]
            decoder_input = state.decoder_input

            decoder_input_embedded_promise = self.embedding.forward_later(decoder_input)
            return encoder_output, decoder_hidden, decoder_input_embedded_promise

        def decode_gru(state: StateTeacherForcing, previous_output):
            encoder_output, decoder_hidden, decoder_input_embedded_promise = previous_output
            decoder_input_embedded = decoder_input_embedded_promise.result
            return encoder_output, self.decoder_gru.forward_later(decoder_input_embedded, decoder_hidden)

        def attention(state: StateTeacherForcing, previous_output):
            encoder_output, decoder_hidden_promise = previous_output
            decoder_output, decoder_hidden = decoder_hidden_promise.result
            return self.att.forward_later(decoder_output, encoder_output)

        def out(state: StateTeacherForcing, previous_output):
            attention_promise = previous_output
            attention_output, _ = attention_promise.result
            output_promise = self.out.forward_later(attention_output)
            return output_promise

        def calc_loss(state: StateTeacherForcing, previous_output):
            output_promise = previous_output
            state.loss = self.loss(output_promise.result, state.target_seq)
            return None

        result_states = ForEachState(states).apply(
            Do(embed)
                .Then(encode)
                .Then(embed_decoder_input)
                .Then(decode_gru)
                .Then(attention)
                .Then(out)
                .Then(calc_loss)
        ).states


        return sum([state.loss for state in result_states]) / len(states)


def train(input_seq_batch, target_seq_batch, model, optimizer, clip):

    # Zero gradients
    optimizer.zero_grad()

    # Initialize variables
    losses = []

    loss = model.forward_teacher_forcing(input_seq_batch, target_seq_batch)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    optimizer.step()
    losses.append(float(loss))

    return sum(losses) / len(losses)


def trainIters(voc, pairs, model, optimizer, n_iteration, batch_size, print_every, clip):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    start_time = time.time()

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, target_variable = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, target_variable, model, optimizer, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; time: {:.1f}, Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, time.time() - start_time, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0


def main():
    create_formatted_data_file()
    voc, pairs = loadPrepareData()

    # Configure models
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    print('Building model ...')
    model = GRUSeq2SeqWithAttention(voc.num_words, hidden_size, encoder_n_layers, decoder_n_layers, dropout)

    # Use appropriate device
    model = model.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    learning_rate = 0.0001
    n_iteration = 4000
    print_every = 1

    # Ensure dropout layers are in train mode
    model.train()

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
    trainIters(voc, pairs, model, optimizer, n_iteration, batch_size, print_every, clip)


if __name__ == "__main__":
    main()
