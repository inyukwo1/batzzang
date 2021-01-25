from typing import Any, List
from abc import ABC, abstractmethod
import torch
from torch import nn
import math
from .tensor_promise import TensorPromise
from .utils import assert_dim, stack_sequential_tensor_with_mask
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F


class LazyModule(ABC):
    waiting_lazy_modules: List["LazyModule"] = []

    def __init__(self):
        self.later_buffer = []
        self.promises = []
        self.done_buffer = []

    @classmethod
    def wait_all(cls):
        for lazy_module in LazyModule.waiting_lazy_modules:
            lazy_module.compute()
            for idx in range(len(lazy_module.later_buffer)):
                lazy_module.promises[idx].result = lazy_module.done_buffer[idx]
            lazy_module.later_buffer = []
            lazy_module.promises = []
            lazy_module.done_buffer = []
        LazyModule.waiting_lazy_modules = []

    def forward_later(self, *inputs) -> TensorPromise:
        if self not in LazyModule.waiting_lazy_modules:
            LazyModule.waiting_lazy_modules.append(self)
        self.assert_input(*inputs)
        self.later_buffer.append(inputs)
        appended_index = len(self.later_buffer) - 1
        promise = TensorPromise(self, appended_index)
        self.promises.append(promise)
        return promise

    def assert_input(self, *inputs):
        pass

    @abstractmethod
    def compute(self) -> None:
        pass


class LazyEmbedding(nn.Module, LazyModule):
    def __init__(self, num_words, hidden_size, dropout):
        super(LazyEmbedding, self).__init__()
        LazyModule.__init__(self)
        self.module = nn.Embedding(num_words, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_words = num_words
        self.hidden_size = hidden_size

    def assert_input(self, *inputs):
        pass

    def compute(self):
        tensor_list = [inputs[0] for inputs in self.later_buffer]
        if len(tensor_list[0].size()) == 1:
            tensor_length = [len(item) for item in tensor_list]
            stacked_tensors = torch.zeros(
                len(tensor_list), max(tensor_length)
            ).long().cuda()
            for idx, _tensor in enumerate(tensor_list):
                stacked_tensors[idx][: len(_tensor)] = _tensor

            computed_tensors = self.dropout(self.module(stacked_tensors))

            # Split
            self.done_buffer = [
                computed_tensor[:length]
                for length, computed_tensor in zip(tensor_length, computed_tensors)
            ]
        elif len(tensor_list[0].size()) == 0:
            stacked_tensors = torch.zeros(
                len(tensor_list)
            ).long().cuda()
            for idx, _tensor in enumerate(tensor_list):
                stacked_tensors[idx] = _tensor

            computed_tensors = self.dropout(self.module(stacked_tensors))

            # Split
            self.done_buffer = [computed_tensor for computed_tensor in computed_tensors]

        else:
            raise Exception("currently we do not support more than two dimension")


class LazyLinear(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyLinear, self).__init__()
        LazyModule.__init__(self)
        self.module = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def assert_input(self, *inputs):
        [tensor] = inputs
        assert isinstance(tensor, torch.Tensor)
        if len(tensor.size()) == 1:
            assert_dim([self.in_dim], tensor)
        elif len(tensor.size()) == 2:
            assert_dim([None, self.in_dim], tensor)

    def compute(self):
        tensor_list = [inputs[0] for inputs in self.later_buffer]
        if len(tensor_list[0].size()) == 2:
            tensor_length = [len(item) for item in tensor_list]
            stacked_tensors = torch.zeros(
                len(tensor_list), max(tensor_length), tensor_list[0].shape[-1]
            ).cuda()
            for idx, _tensor in enumerate(tensor_list):
                stacked_tensors[idx][: len(_tensor)] = _tensor

            computed_tensors = self.module(stacked_tensors)

            # Split
            self.done_buffer = [
                computed_tensor[:length]
                for length, computed_tensor in zip(tensor_length, computed_tensors)
            ]
        elif len(tensor_list[0].size()) == 1:
            stacked_tensors = torch.zeros(
                len(tensor_list), tensor_list[0].shape[-1]
            ).cuda()
            for idx, _tensor in enumerate(tensor_list):
                stacked_tensors[idx] = _tensor

            computed_tensors = self.module(stacked_tensors)

            # Split
            self.done_buffer = [computed_tensor for computed_tensor in computed_tensors]
        else:
            raise Exception("currently we do not support more than two dimension")


class LazyLSTMCell(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyLSTMCell, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module = nn.LSTMCell(in_dim, out_dim)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        input_list = []
        hidden_list = []
        cell_list = []
        for item in self.later_buffer:
            input_list.append(item[0])
            hidden_list.append(item[1][0])
            cell_list.append(item[1][1])

        # Stacked
        stacked_input = torch.stack(input_list)
        stacked_hidden = torch.stack(hidden_list)
        stacked_cell = torch.stack(cell_list)

        next_hid, next_cell = self.module(stacked_input, (stacked_hidden, stacked_cell))

        self.done_buffer = [(hid, cell) for hid, cell in zip(next_hid, next_cell)]


class LazyLSTM(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim, n_layers, bidirectional=True):
        super(LazyLSTM, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module = nn.LSTM(
            in_dim, out_dim, n_layers, batch_first=True, bidirectional=bidirection
        )
        self.bidirectional = bidirectional

    def assert_intput(self, *inputs):
        pass

    def compute(self):
        input_list = [item[0] for item in self.later_buffer]

        # Stack
        stacked_input, input_mask = stack_sequential_tensor_with_mask(input_list)

        # Sort
        input_len = [len(item) for item in input_list]
        sorted_len, sorted_indices = torch.tensor(input_len).sort(0, descending=True)
        sorted_data = stacked_input[sorted_indices]

        # Pass
        packed_data = pack_padded_sequence(sorted_data, sorted_len, batch_first=True)
        packed_output, (h_n, c_n) = self.module(packed_data)
        packed_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        num_layers_direction, batch, hidden_size = list(h_n.size())


        # Unsort
        new_indices = list(range(len(input_len)))
        new_indices.sort(key=lambda k: sorted_indices[k])
        encoded_data = packed_output[new_indices]
        if self.bidirectional:
            h_n = h_n.view(num_layers_direction // 2, 2, batch, hidden_size)
            c_n = c_n.view(num_layers_direction // 2, 2, batch, hidden_size)
            last_state = h_n[:, :, new_indices, :]
            last_cell = c_n[:, :, new_indices, :]

            # spread inputs
            self.done_buffer = [
                (encoded_data[idx][:input_len[idx]], last_state[:, :, idx, :], last_cell[:, :, idx, :])
                for idx in range(len(input_len))
            ]
        else:
            last_state = h_n[:, new_indices, :]
            last_cell = c_n[:, new_indices, :]

            # spread inputs
            self.done_buffer = [
                (encoded_data[idx][:input_len[idx]], last_state[:, idx, :], last_cell[:, idx, :])
                for idx in range(len(input_len))
            ]


class LazyGRUCell(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyGRUCell, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module = nn.GRUCell(
            in_dim, out_dim
        )

    def assert_intput(self, *inputs):
        pass

    def compute(self):
        input_list = []
        hidden_list = []
        for item in self.later_buffer:
            input_list.append(item[0])
            hidden_list.append(item[1])

        # Stacked
        stacked_input = torch.stack(input_list)
        stacked_hidden = torch.stack(hidden_list)

        next_hid = self.module(stacked_input, stacked_hidden)

        self.done_buffer = [hid for hid in next_hid]


class LazyGRU(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim, n_layers, bidirection=True):
        super(LazyGRU, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module = nn.GRU(
            in_dim, out_dim, n_layers, batch_first=True, bidirectional=bidirection
        )
        self.bidirection = bidirection

    def assert_intput(self, *inputs):
        pass

    def compute(self):
        input_list = [item[0] for item in self.later_buffer]

        # Stack
        stacked_input, input_mask = stack_sequential_tensor_with_mask(input_list)

        # Sort
        input_len = [len(item) for item in input_list]
        sorted_len, sorted_indices = torch.tensor(input_len).sort(0, descending=True)
        sorted_data = stacked_input[sorted_indices]

        # Pass
        packed_data = pack_padded_sequence(sorted_data, sorted_len, batch_first=True)
        packed_output, h_n = self.module(packed_data)
        num_layers_direction, batch, hidden_size = list(h_n.size())
        if self.bidirection:
            h_n = h_n.view(num_layers_direction // 2, 2, batch, hidden_size)
        packed_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Unsort
        new_indices = list(range(len(input_len)))
        new_indices.sort(key=lambda k: sorted_indices[k])
        encoded_data = packed_output[new_indices]
        if self.bidirection:
            last_state = h_n[:, :, new_indices, :]
            # spread inputs
            self.done_buffer = [
                (encoded_data[idx][:input_len[idx]], last_state[:, :, idx, :])
                for idx in range(len(input_len))
            ]
        else:
            last_state = h_n[:, new_indices, :]
            # spread inputs
            self.done_buffer = [
                (encoded_data[idx][:input_len[idx]], last_state[:, idx, :])
                for idx in range(len(input_len))
            ]


class LazyTransformerDecoder(nn.Module, LazyModule):
    def __init__(self, d_model, nhead, dim_feedforward, layer_num, dropout=0.1, activation='relu'):
        super(LazyTransformerDecoder, self).__init__()
        LazyModule.__init__(self)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation)
        self.module = nn.TransformerDecoder(decoder_layer, num_layers=layer_num)
        self.in_dim = d_model
        self._init_positional_embedding()

    def _init_positional_embedding(self, dropout=0.1, max_len=500):
        d_model = self.in_dim

        self.pos_dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def _pos_encode(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.pos_dropout(x)

    def assert_input(self, *inputs):
        tgt, mem = inputs
        assert_dim([None, self.in_dim], tgt)
        assert_dim([None, self.in_dim], mem)

    def compute(self):
        tgt_list: List[torch.Tensor] = []
        mem_list: List[torch.Tensor] = []
        for tgt, mem in self.later_buffer:
            tgt_list.append(tgt)
            mem_list.append(mem)
        stacked_tgt, tgt_padding_mask = stack_sequential_tensor_with_mask(tgt_list)
        stacked_mem, mem_padding_mask = stack_sequential_tensor_with_mask(mem_list)
        stacked_tgt_batch_second = stacked_tgt.transpose(0, 1)
        stacked_mem_batch_second = stacked_mem.transpose(0, 1)
        pos_encoded_tgt_batch_second = self._pos_encode(stacked_tgt_batch_second)

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(None, list(stacked_tgt_batch_second.size())[0]).cuda()
        out_batch_first = self.module(
            pos_encoded_tgt_batch_second,
            stacked_mem_batch_second,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=mem_padding_mask,
        ).transpose(0, 1)
        self.done_buffer = [
            out_batch_first[idx, : len(tgt_list[idx])]
            for idx in range(len(self.later_buffer))
        ]


class LazyAttention(nn.Module, LazyModule):
    def __init__(self, dimensions, attention_type='general'):
        super(LazyAttention, self).__init__()
        LazyModule.__init__(self)
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def assert_input(self, *inputs):
        pass

    def compute(self):
        query_list: List[torch.Tensor] = []
        context_list: List[torch.Tensor] = []
        for q, c in self.later_buffer:
            query_list.append(q)
            context_list.append(c)
        query, q_mask = stack_sequential_tensor_with_mask(query_list)
        context, c_mask = stack_sequential_tensor_with_mask(context_list)

        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        attention_scores.data.masked_fill_(q_mask.unsqueeze(2).bool(), float("-inf"))
        attention_scores.data.masked_fill_(c_mask.unsqueeze(1).bool(), float("-inf"))
        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        attention_weights.data.masked_fill_(c_mask.unsqueeze(1).bool(), 0.)

        attention_weights.data.masked_fill_(q_mask.unsqueeze(2).bool(), 0.)
        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)
        self.done_buffer = [
            (output[i, :len(query_list[i])], attention_weights[i, :len(query_list[i])])
            for i in range(len(output))
        ]


class LazyPointerNet(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyPointerNet, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.pass_linear = nn.Linear(in_dim, out_dim, bias=False)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        query_list = [item[0] for item in self.later_buffer]
        key_list = [item[1] for item in self.later_buffer]

        stacked_query, query_mask = stack_sequential_tensor_with_mask(query_list)
        stacked_key = torch.stack(key_list)

        encoded_query = self.pass_linear(stacked_query)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        weights = torch.bmm(encoded_query, stacked_key.unsqueeze(2)).squeeze(2)

        weights.data.masked_fill_(query_mask.bool(), float("-inf"))
        probs = torch.log_softmax(weights, dim=-1)

        self.done_buffer = [item for item in probs]


class LazyMemoryPointerNet(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyMemoryPointerNet, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pass_gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.col_linear = nn.Linear(in_dim, out_dim, bias=False)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        encoded_col_list = [item[0] for item in self.later_buffer]
        att_vec_list = [item[1] for item in self.later_buffer]
        col_memory_mask = [item[2] for item in self.later_buffer]

        stacked_col, col_mask = stack_sequential_tensor_with_mask(encoded_col_list)
        stacked_att = torch.stack(att_vec_list)
        stacked_mem_mask, _ = stack_sequential_tensor_with_mask(col_memory_mask)

        # Create gate
        gate = self.pass_gate(stacked_att)

        encoded_col_stack = self.col_linear(stacked_col)

        weights = torch.bmm(encoded_col_stack, stacked_att.unsqueeze(2)).squeeze(-1)
        one = weights * stacked_mem_mask * gate
        two = weights * (1 - stacked_mem_mask) * (1 - gate)
        total = one + two

        total.data.masked_fill_(col_mask.bool(), float("-inf"))
        probs = torch.log_softmax(total, dim=-1)

        self.done_buffer = [item for item in probs]


class LazyMultiHeadAttention(nn.Module, LazyModule):
    def __init__(self, nhead, q_dim, v_dim, dropout=0.1):
        super(LazyMultiHeadAttention, self).__init__()
        LazyModule.__init__(self)
        self.q_linear = torch.nn.Linear(q_dim, v_dim)
        self.k_linear = torch.nn.Linear(v_dim, v_dim)
        self.v_linear = torch.nn.Linear(v_dim, v_dim)
        self.multi_head_attention = nn.MultiheadAttention(v_dim, nhead, dropout=dropout)
        self.out_linear = torch.nn.Linear(v_dim, v_dim)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        query, query_mask = stack_sequential_tensor_with_mask([item[0] for item in self.later_buffer])
        key, key_mask = stack_sequential_tensor_with_mask([item[1] for item in self.later_buffer])
        value, _ = stack_sequential_tensor_with_mask([item[1] for item in self.later_buffer])

        query = self.q_linear(query).transpose(0, 1)
        key = self.k_linear(key).transpose(0, 1)
        value = self.v_linear(value).transpose(0, 1)

        new_representation, attn_weights = self.multi_head_attention(query, key, value, key_padding_mask=key_mask)
        out_representation = self.out_linear(new_representation.transpose(0, 1))

        self.done_buffer = [item for item in out_representation]
