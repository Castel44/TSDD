import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from src.utils.decorators import deprecated

import numpy as np


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, x):
        norm = torch.clamp(x.norm(2, -1, keepdim=True), min=1)
        x = x / norm
        # x = x/torch.norm(x, 2)
        return x


class MetaAE(nn.Module):
    def __init__(self, name='AE'):
        super(MetaAE, self).__init__()

        self.encoder = None
        self.decoder = None

        self.name = name

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, x):
        # Not Used for now
        emb = self.encoder(x)
        emb = emb / torch.sqrt(torch.sum(emb ** 2, 2, keepdim=True))
        return emb

    def get_name(self):
        return self.name


#########################################################################################################
# CONVOLUTIONAL AUTOENCODER
#########################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, dropout=0.2, normalization='none'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.conv, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        self.convtraspose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                               output_padding=output_padding,
                                               padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.convtraspose, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        num_blocks = len(num_channels)
        layers = []
        for i in range(num_blocks):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dropout=dropout,
                          normalization=normalization)]

        self.network = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv1d(num_channels[-1], embedding_dim, 1)

    def forward(self, x):
        x = self.network(x.transpose(2, 1))
        x = F.max_pool1d(x, kernel_size=x.data.shape[2])
        x = self.conv1x1(x)
        return x


def conv_out_len(seq_len, ker_size, stride, padding, dilation, stack):
    for _ in range(stack):
        seq_len = int((seq_len + 2*padding - dilation*(ker_size-1)-1)/stride +1)
    return seq_len


class ConvDecoder(nn.Module):
    def __init__(self, embedding_dim, num_channels, seq_len, out_dimension, kernel_size, stride=2, padding=0,
                 dropout=0.2, normalization='none'):
        super().__init__()

        num_channels = num_channels[::-1]
        num_blocks = len(num_channels)

        self.compressed_len = conv_out_len(seq_len, kernel_size, stride, padding, 1, num_blocks)

        if stride > 1:
            output_padding = []
            seq = seq_len
            for _ in range(num_blocks):
                output_padding.append(seq % 2)
                seq = conv_out_len(seq, kernel_size, stride, padding, 1, 1)

            def convt_shape(seq_len, ker_size, stride, padding, dilation, out_pad):
                return (seq_len -1)*stride - 2*padding + dilation*(ker_size-1)+out_pad+1

            # bit flip
            if kernel_size % 2 == 1:
                output_padding = [1 -x for x in output_padding[::-1]]
            else:
                output_padding = output_padding[::-1]
        else:
            output_padding = [0]*num_blocks

        layers = []
        for i in range(num_blocks):
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvTransposeBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding[i], dropout=dropout, normalization=normalization)]
        self.network = nn.Sequential(*layers)
        self.upsample = nn.Linear(1, self.compressed_len)
        self.conv1x1 = nn.Conv1d(num_channels[-1], out_dimension, 1)

    def forward(self, x):
        x = self.upsample(x)
        # x = F.pad(x, pad=(self.seq_len-1, 0), mode='constant', value=0)
        x = self.network(x)
        x = self.conv1x1(x)
        return x.transpose(2, 1)


class CNNAE(MetaAE):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, kernel_size, dropout,
                 normalization=None, stride=2, padding=0, name='CNN_AE'):
        super(CNNAE, self).__init__(name=name)

        self.encoder = ConvEncoder(input_size, num_filters, embedding_dim, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dropout=dropout, normalization=normalization)
        self.decoder = ConvDecoder(embedding_dim, num_filters, seq_len, input_size, kernel_size, stride=stride,
                                   padding=padding, dropout=dropout, normalization=normalization)


#########################################################################################################
# TEMPORAL CONVOLUTIONAL AUTOENCODER
#########################################################################################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 normalization='none', activation=nn.ReLU()):
        assert normalization in ['weight', 'batch', 'layer', 'none'], 'Only weight, batch or layer normalization'

        super(TemporalBlock, self).__init__()
        if normalization == 'weight':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)

        self.chomp1 = Chomp1d(padding)

        if normalization == 'batch':
            self.norm1 = nn.BatchNorm1d(n_outputs)
        elif normalization == 'layer':
            self.norm1 = nn.LayerNorm(n_outputs)
        else:
            self.norm1 = None

        self.act1 = activation
        self.dropout1 = nn.Dropout(dropout)

        if normalization == 'weight':
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
        else:
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)

        self.chomp2 = Chomp1d(padding)

        if normalization == 'batch':
            self.norm2 = nn.BatchNorm1d(n_outputs)
        elif normalization == 'layer':
            self.norm2 = nn.LayerNorm(n_outputs)
        else:
            self.norm2 = None

        self.act2 = activation
        self.dropout2 = nn.Dropout(dropout)

        self.layers = [self.conv1, self.chomp1, self.norm1, self.act1, self.dropout1,
                       self.conv2, self.chomp2, self.norm2, self.act2, self.dropout2]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.act3 = activation
        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act3(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, out_dimension, kernel_size=2, dropout=0.2, activation=nn.ReLU(),
                 normalization=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout, activation=activation,
                                     normalization=normalization)]

        self.network = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv1d(num_channels[-1], out_dimension, 1)

    def forward(self, x):
        x = self.network(x)
        x = self.conv1x1(x)
        return x


class TemporalEncoder(TemporalConvNet):
    def __init__(self, *args, **kwargs):
        super(TemporalEncoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.network(x.transpose(2, 1))
        x = x[:, :, -1]
        return self.conv1x1(x[:, :, np.newaxis])


class TemporalDecoder(TemporalConvNet):
    def __init__(self, seq_len, *args, **kwargs):
        super(TemporalDecoder, self).__init__(*args, **kwargs)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.repeat(1, 1, self.seq_len)
        # x = F.pad(x, pad=(self.seq_len-1, 0), mode='constant', value=0)
        x = self.network(x)
        x = self.conv1x1(x)
        return x.transpose(2, 1)


class TCNAE(MetaAE):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, num_stack, kernel_size, dropout,
                 normalization=None,
                 hidd_act=nn.ReLU(), name='TCN_AE'):
        super(TCNAE, self).__init__(name=name)
        self.num_channels = [num_filters] * num_stack
        # self.receptive_field = kernel_size * num_stack * 2 ** (num_stack - 1) + 1

        self.encoder = TemporalEncoder(input_size, self.num_channels, embedding_dim, kernel_size=kernel_size,
                                       dropout=dropout, normalization=normalization, activation=hidd_act)
        self.decoder = TemporalDecoder(seq_len, embedding_dim, self.num_channels, input_size, kernel_size=kernel_size,
                                       dropout=dropout, normalization=normalization, activation=hidd_act)
        # self.encoder = nn.Sequential(self.encoder, NormLayer())


#########################################################################################################
# RECURRENT AUTOENCODER
#########################################################################################################
class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim, embedding_dim=64, n_layers=1):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True
        )

        self.embedding = nn.Linear(self.hidden_dim, self.embedding_dim)

    def forward(self, x):
        # x has shape, (BS, seq_len, input_size)
        _, (hidden_n, _) = self.rnn(x)
        h = hidden_n[-1].view(-1, self.hidden_dim)  # batch first (BS, embedding_dim)
        out = self.embedding(h)
        return out


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, hidden_dim=16, n_features=1, n_layers=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = hidden_dim, n_features
        self.n_layers = n_layers

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True
        )
        #self.upsample = nn.Linear(1, self.seq_len)
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        #x = self.upsample(x.unsqueeze(2)).permute(0, 2, 1)

        # x = x.reshape((-1, self.seq_len, self.input_dim*self.n_features))
        # x, (hidden_n, cell_n) = self.rnn1(x)
        x, (_, _) = self.rnn(x)
        # x = x.view(-1, self.seq_len, self.hidden_dim)
        # x = self.output_layer(x).view(-1, self.seq_len)
        x = self.output_layer(x)
        return x


class LSTMAE(MetaAE):
    # TODO: add BN and DR
    # TODO: check init wheight
    # TODO: check rolling prediction and clean up the code
    def __init__(self, seq_len_out, n_features, n_layers=1, hidden_dim=128, embedding_dim=64, name='LSTM_AE'):
        super(LSTMAE, self).__init__(name=name)
        self.encoder = Encoder(n_features, hidden_dim, embedding_dim, n_layers)
        self.decoder = Decoder(seq_len_out, embedding_dim, hidden_dim, n_features, n_layers)


#########################################################################################################
# MLP AUTOENCODER
#########################################################################################################
class MLP(nn.Module):
    @staticmethod
    def block(in_size, out_size, activation=nn.ReLU(), normalization='none', p=0.15):
        if normalization == 'batch':
            norm = nn.BatchNorm1d(out_size)
        elif normalization == 'layer':
            norm = nn.LayerNorm(out_size)
        else:
            norm = None

        layers = [nn.Linear(in_size, out_size), norm, activation, nn.Dropout(p)]
        return nn.Sequential(*[x for x in layers if x is not None])

    def __init__(self, input_shape: int, output_shape: int, hidden_neurons: list, hidd_act=nn.ReLU(),
                 dropout: float = 0.0, normalization: str = 'none'):
        super(MLP, self).__init__()

        self.model_size = [input_shape, *hidden_neurons]
        self.norm = normalization
        self.p = dropout
        self.hidd_act = hidd_act

        blocks = [self.block(in_f, out_f, activation=self.hidd_act, normalization=self.norm, p=self.p) for in_f, out_f
                  in zip(self.model_size, self.model_size[1::])]
        self.model = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden_neurons[-1], output_shape)

    def forward(self, x):
        # TODO: fix output activation
        x = self.model(x)
        return self.out(x)


class MLPAE(MetaAE):
    """
    Wrapper to MLP to create an AutoEncoder
    """

    def __init__(self, input_shape: int, embedding_dim: int, hidden_neurons: list, hidd_act=nn.ReLU(),
                 dropout: float = 0.0, normalization: str = 'none', name='MLP_AE'):
        super(MLPAE, self).__init__(name=name)
        self.encoder = MLP(input_shape, embedding_dim, hidden_neurons, hidd_act, dropout=dropout,
                           normalization=normalization)
        self.decoder = MLP(embedding_dim, input_shape, hidden_neurons[::-1], hidd_act, dropout=dropout,
                           normalization=normalization)

        # self.encoder = nn.Sequential(self.encoder, NormLayer())
