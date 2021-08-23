import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import numpy as np


class MetaModel(nn.Module):
    def __init__(self, ae, classifier, n_out, name='network'):
        super(MetaModel, self).__init__()

        self.encoder = ae.encoder
        self.classifier = nn.ModuleList([copy.deepcopy(classifier) for i in range(n_out)])
        self.n_out = n_out
        self.name = name

    def forward(self, x):
        x_enc = self.encoder(x).squeeze()
        x_out = [self.classifier[i](x_enc) for i in range(self.n_out)]
        return torch.stack(x_out, dim=-1).squeeze(-1)
        #return x_out

    def get_name(self):
        return self.name


class AEandClass(MetaModel):
    def __init__(self, ae, **kwargs):
        super(AEandClass, self).__init__(ae, **kwargs)
        self.decoder = ae.decoder

    def forward(self, x):
        x_enc = self.encoder(x)
        xhat = self.decoder(x_enc)
        x_out = [self.classifier[i](x_enc.squeeze(-1)) for i in range(self.n_out)]
        return xhat, torch.stack(x_out, dim=-1).squeeze(-1), x_enc


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


if __name__ == '__main__':
    import torch
    from src.models.MLP import MLP
    from src.models.AEs import MLPAE, LSTMAE, TCNAE
    from torchviz import make_dot

    input_dim = 20
    embedding_dim = 2
    seq_len = 15
    neurons = [10, 5]
    classes = 3
    n_out = 2

    x = torch.randn((5, seq_len, input_dim))
    y = torch.randint(classes, (5,))

    # model_ae = MLPAE(input_dim, embedding_dim, neurons)
    # model_ae = Seq2Seq(seq_len, seq_len, input_dim, n_layers=2, embedding_dim=embedding_dim)
    nonlinclassifier = NonLinClassifier(embedding_dim, classes, norm='none')
    linclassifier = LinClassifier(embedding_dim, classes)
    
    model_ae = TCNAE(input_dim, 3, embedding_dim, 2, 3, 0, 'none')
    model = MetaModel(ae=model_ae, classifier=nonlinclassifier, n_out=n_out)
    model_full = AEandClass(ae=model_ae, classifier=nonlinclassifier, n_out=n_out)
    # model.classifier[1]

    y_ae = model_ae(x)
    yhat = model(x)
    y_full = model_full(x)

    # make_dot(yhat, params=dict(list(model.named_parameters()))).render("classifier", directory='torchviz', format="png")
    # make_dot(out_ae, params=dict(list(model_ae.named_parameters()))).render("AE",  directory='torchviz', format="png")

    input_names = ['Input']
    output_names = ['yhat'] * n_out
    torch.onnx.export(model, x, 'model.onnx', export_params=False, input_names=input_names, output_names=output_names)
    torch.onnx.export(model_ae, x, 'model_ae.onnx', export_params=False, input_names=input_names,
                      output_names=['output'])
    torch.onnx.export(model_full, x, 'model_full.onnx', export_params=False, input_names=input_names,
                      output_names=['recons'] + output_names)
