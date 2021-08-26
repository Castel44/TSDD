import torch.nn as nn


#########################################################################################################
# Meta Class
#########################################################################################################
class MetaClass(nn.Module):
    def __init__(self, name='model'):
        super(MetaClass, self).__init__()

        self.encoder = None
        self.classifier = None

        self.name = name

    def forward(self, x):
        x_enc = self.encoder(x).squeeze()
        x_out = self.classifier(x_enc)
        return x_out

    def get_embedding(self, x):
        # Not Used for now
        emb = self.encoder(x)
        emb = emb.squeeze()
        # emb = emb / torch.sqrt(torch.sum(emb ** 2, 2, keepdim=True))
        return emb

    def get_name(self):
        return self.name


#########################################################################################################
# MLP
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


#########################################################################################################
# Output Classifiers
#########################################################################################################
class NonLinClassifier(nn.Module):
    def __init__(self, d_in, n_class, d_hidd=16, activation=nn.ReLU(), dropout=0.1, norm='batch'):
        """
        norm : str : 'batch' 'layer' or None
        """
        super(NonLinClassifier, self).__init__()

        self.dense1 = nn.Linear(d_in, d_hidd)

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(d_hidd)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(d_hidd)
        else:
            self.norm = None

        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_hidd, n_class)

        self.layers = [self.dense1, self.norm, self.act, self.dropout, self.dense2]
        self.net = nn.Sequential(*[x for x in self.layers if x is not None])

    def forward(self, x):
        out = self.net(x)
        return out


class LinClassifier(nn.Module):
    def __init__(self, d_in, n_class):
        super(LinClassifier, self).__init__()
        self.dense = nn.Linear(d_in, n_class)

    def forward(self, x):
        out = self.dense(x)
        return out


#########################################################################################################
# Model
#########################################################################################################
class MetaMLP(MetaClass):
    """
    Wrapper to MLP to create an AutoEncoder
    """

    def __init__(self, input_shape: int, embedding_dim: int, hidden_neurons: list, n_class: int, hidd_act=nn.ReLU(),
                 dropout: float = 0.0, normalization: str = 'none', name='MLP'):
        super(MetaMLP, self).__init__(name=name)
        self.encoder = MLP(input_shape, embedding_dim, hidden_neurons, hidd_act, dropout=dropout,
                           normalization=normalization)
        self.classifier = LinClassifier(embedding_dim, n_class)
