import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnEncoder(nn.Module):
    def __init__(self, n_words, n_factors, n_hidden, n_layers, bidirectional=False):
        super().__init__()
        self.e = nn.Embedding(n_words, n_factors)
        self.gru = GRU(n_factors, n_hidden, n_layers, bidirectional=bidirectional)

    def forward(self, *args):
        out = self.e(args[0])
        if len(args) == 1:
            out, hidd = self.gru(out)
        elif len(args) == 2:
            out, hidd = self.gru(out, args[1])
        return out, hidd


class AttnDecoder(nn.Module):
    def __init__(self, n_words, n_factors, n_hidden_dec, n_hidden_enc, n_dir_enc, n_allign, n_layers,
                 n_l):

        super().__init__()

        if n_l % 2 != 0:
            raise ValueError
        self.t_ids = self.get_indices(n_l * 2)

        self.n_layers = n_layers
        self.n_hidden = n_hidden_dec
        self.n_words = n_words

        self.emb = nn.Embedding(n_words, n_factors)
        self.align = nn.ModuleList([nn.Linear((n_dir_enc * n_hidden_enc + n_hidden_dec), n_allign),
                                    nn.Linear(n_allign, 1)])
        self.gru = GRU((n_factors + n_dir_enc * n_hidden_enc), n_hidden_dec, n_layers)
        self.t_hat = nn.Linear(n_hidden_dec + n_factors + (n_hidden_enc*n_dir_enc), 2 * n_l)
        self.out = nn.Linear(n_l, n_words)

    def forward(self, inp, state, prev_word):
        """
        Feed forward
        Input:
            inp - (seq_len,bs,n_hidd_enc*n_dir_enc)
            state - (n_layers,bs,n_hidden_dec)
            prev_word - (1,bs)
        Output:
            out - (1,bs,n_words)
            hidd - (n_layers,bs,n_hidden)
        """
        prev_word = self.emb(prev_word)
        bs = inp.size(1)

        if state is None:
            state = torch.zeros(self.n_layers, bs, self.n_hidden).cuda()

        c = self.context(state[0], inp)[None, :]

        out, hidd = self.gru(torch.cat((prev_word, c), -1), state)

        t_hat = self.t_hat(torch.cat((out,prev_word,c),-1))
        t = self.compute_t(t_hat)

        out = self.out(t)
        out = F.log_softmax(out, -1)

        return out, hidd

    def context(self, state, inp):
        """
        Compute context
        Input:
            state - (bs,n_hidden_dec)
            inp - (seq_len,bs,n_hidd_enc*n_dir_enc)
        Output:
            c - (1,bs,n_hidd_enc*n_dir_enc)
        """
        state_expd = state.expand(len(inp), *state.size())

        c = self.align[0](torch.cat((inp, state_expd), -1))
        c = self.align[1](torch.tanh(c))
        c = F.softmax(c, 0)
        c = (c * inp).sum(0)

        return c

    def compute_t(self, t_hat):
        t1 = t_hat[...,self.t_ids[0]]
        t2 = t_hat[...,self.t_ids[1]]
        t = torch.max(t1,t2)
        return t

    @staticmethod
    def get_indices(n_l):
        odd, even = [], []
        for n in range(n_l):
            if n % 2 == 0:
                even.append(n)
            else:
                odd.append(n)
        return even, odd

class GRU(nn.Module):
    """
    Basic GRU, IO dimensions correspond to PyTorch implementation
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
        self.gru_f = MonoGRU(n_input, n_hidden, n_layers)
        self.gru_b = MonoGRU(n_input, n_hidden, n_layers)

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

        out_f, new_hidd_f = self.gru_f(inps_f, hidd_f)
        out_b, new_hidd_b = self.gru_b(inps_b, hidd_b)

        out = torch.cat((out_f, out_b), -1)
        new_hidd = torch.cat((new_hidd_f, new_hidd_b), 0)

        return out, new_hidd

    def init_hidden(self, bs):
        return torch.zeros(2 * self.n_layers, bs, self.n_hidden).cuda()


class MonoGRU(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.state = nn.ModuleList([nn.Linear(n_input + n_hidden, n_hidden)]
                                   + [nn.Linear(2 * n_hidden, n_hidden) for _ in range(n_layers-1)])
        self.reset = nn.ModuleList([nn.Linear(n_input + n_hidden, n_hidden)]
                                   + [nn.Linear(2 * n_hidden, n_hidden) for _ in range(n_layers-1)])
        self.update = nn.ModuleList([nn.Linear(n_input + n_hidden, n_hidden)]
                                    + [nn.Linear(2 * n_hidden, n_hidden) for _ in range(n_layers-1)])

    def forward(self, *args):
        inps = args[0]
        bs = inps.size(1)
        if len(args) == 1:
            hidd = self.init_hidden(bs)
        elif len(args) == 2:
            hidd = args[1]

        outs = torch.zeros(inps.size(0), *hidd.size()[1:]).cuda()
        hidds = self.init_hidden(bs)

        for i, hidd_l in enumerate(hidd):
            for j, inp in enumerate(inps):
                r = torch.sigmoid(self.reset[i](torch.cat((inp, hidd_l), -1)))
                z = torch.sigmoid(self.update[i](torch.cat((inp, hidd_l), -1)))
                new_s = torch.tanh(self.state[i](torch.cat((inp, (r * hidd_l)), -1)))
                hidd_l = z * hidd_l + (1 - z) * new_s
                outs[j] = hidd_l
            hidds[i] = outs[-1]
            inps = outs.clone()

        return outs, hidds

    def init_hidden(self, bs):
        return torch.zeros(self.n_layers, bs, self.n_hidden).cuda()
