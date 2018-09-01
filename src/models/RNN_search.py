import torch
import torch.nn as nn


class GRU(nn.Module):
    """
    - Basic GRU, IO dimensions correspond to PyTorch implementation
    """

    def __init__(self, n_input, n_hidden, n_layers, bidirectional=False):
        super().__init__()
        if not bidirectional:
            self.gru = MonoGRU(n_input, n_hidden, n_layers)
        else:
            self.gru = BiGRU(n_input, n_hidden, n_layers)

    def forward(self, *args):
        return self.gru(*args)


class BiGRU(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.gru = MonoGRU(n_input, n_hidden, n_layers)

    def forward(self, *args):
        # f = forward, b = backward
        inps_f = args[0]
        inps_b = inps_f.flip(0)
        bs = inps_f.size(1)

        if len(args) == 1:
            hidd = self.init_hidden(bs)
        elif len(args) == 2:
            hidd = args[1]

        split_pt = self.n_layers
        hidd_f = hidd[:split_pt]
        hidd_b = hidd[split_pt:]

        out_f, new_hidd_f = self.gru(inps_f, hidd_f)
        out_b, new_hidd_b = self.gru(inps_b, hidd_b)

        out = torch.cat((out_f, out_b), -1)
        new_hidd = torch.cat((new_hidd_f, new_hidd_b), 0)

        return out, new_hidd

    def init_hidden(self, bs):
        return torch.zeros(2 * self.n_layers, bs, self.n_hidden)


class MonoGRU(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.state = nn.ModuleList([nn.Linear(n_input + n_hidden, n_hidden)]
                                   + [nn.Linear(2 * n_hidden, n_hidden) for x in range(n_layers)])
        self.reset = nn.ModuleList([nn.Linear(n_input + n_hidden, n_hidden)]
                                   + [nn.Linear(2 * n_hidden, n_hidden) for x in range(n_layers)])
        self.update = nn.ModuleList([nn.Linear(n_input + n_hidden, n_hidden)]
                                    + [nn.Linear(2 * n_hidden, n_hidden) for x in range(n_layers)])

    def forward(self, *args):
        inps = args[0]
        bs = inps.size(1)
        if len(args) == 1:
            hidd = self.init_hidden(bs)
        elif len(args) == 2:
            hidd = args[1]

        outs = torch.zeros(inps.size(0), *hidd.size()[1:])
        hidds = torch.zeros(hidd.size())

        for i, hidd_l in enumerate(hidd):
            for j, inp in enumerate(inps):
                r = torch.sigmoid(self.reset[i](torch.cat((inp, hidd_l), -1)))
                z = torch.sigmoid(self.update[i](torch.cat((inp, hidd_l), -1)))
                new_s = torch.tanh(self.state[i](torch.cat((inp, (r * hidd_l)), -1)))
                new_hidd = z * hidd_l + (1 - z) * new_s
                outs[j] = new_hidd
            hidds[i] = outs[-1]

        return outs, hidds

    def init_hidden(self, bs):
        return torch.zeros(self.n_layers, bs, self.n_hidden)
