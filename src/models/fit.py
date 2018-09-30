import torch
import random
import math

from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

from src.utils import pad_len_sort_both


class Learner:
    def __init__(self, modules, optimizers, loss_fn, models_directory):
        self.modules = modules
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.models_directory = models_directory

    def train(self, train, valid, epochs, batch_size, plot_every, end_train, end_val, *args):

        for m in self.modules: m.train()
        to_plot = []
        checkpoint = 1

        for e in range(1, epochs + 1):
            train_iter = iter(train)

            example_num = 1
            avg_loss = 0

            for o in self.optimizers: o.zero_grad()

            for _ in tqdm(range(end_train)):

                example = next(train_iter)

                out, trg = self.get_output_target(False, example, *args)

                loss = self.loss_fn(out, trg)
                loss.backward()
                avg_loss += loss.item()

                if example_num % batch_size == 0 or example_num == end_train:
                    for o in self.optimizers:
                        o.step()
                        o.zero_grad()

                if example_num % plot_every == 0:
                    to_plot.append(avg_loss / plot_every)
                    avg_loss = 0

                example_num += 1

            print(f'Train: {self.get_loss(train, end_val, *args)} \n'
                  f'Valid: {self.get_loss(valid, end_val, *args)}')
            for i, m in enumerate(self.modules):
                torch.save(m.state_dict(),f'{self.models_directory}module{i+1}_{checkpoint}.pt')
            checkpoint += 1

        plt.plot(to_plot)

    def get_loss(self, data, end_val, *args):

        for m in self.modules: m.eval()

        avg_loss = 0
        example_num = 1

        for example in data:

            out, trg = self.get_output_target(True, example, *args)

            loss = self.loss_fn(out, trg)
            avg_loss += loss.item()

            if example_num % end_val == 0:
                for m in self.modules: m.train()
                return avg_loss / example_num

            example_num += 1

    # eval refers to context this function is being used in - training stage -> False, evaluation stage -> True
    def get_output_target(self, eval, example, *args):
        raise NotImplementedError


class RNNsearch_Learner(Learner):
    def get_output_target(self, eval, example, src_name, trg_name, trg_vocab_size, teacher_forcing):
        enc, dec = self.modules
        src, trg = getattr(example, src_name), getattr(example, trg_name)

        old_out, _ = enc(src)
        out = torch.zeros(trg.size(0) - 1, trg.size(1), trg_vocab_size).cuda()

        if eval: teacher_forcing = 0
        use_teacher_forcing = True if random.random() < teacher_forcing else False

        prev_word = trg[0][None, :]
        state = None

        for i in range(0, len(trg) - 1):
            out_word, state = dec(old_out, state, prev_word)
            out[i] = out_word
            if use_teacher_forcing:
                prev_word = trg[i + 1][None, :]
            else:
                prev_word = torch.argmax(out_word, dim=-1).detach()

        out = out.permute(1, 2, 0)
        trg = trg[1:].permute(1, 0)

        return out, trg


class LSTM_RNN_Learner(Learner):
    def get_output_target(self, eval, example, src_name, trg_name, trg_vocab_size, teacher_forcing):
        enc, dec = self.modules
        src, trg = getattr(example, src_name), getattr(example, trg_name)

        hidd = enc(src)
        out = torch.zeros(trg.size(0) - 1, trg.size(1), trg_vocab_size).cuda()

        if eval: teacher_forcing = 0
        use_teacher_forcing = True if random.random() < teacher_forcing else False

        if use_teacher_forcing:
            out, _ = dec(trg[:-1], hidd)

        else:
            prev_word = trg[0][None, :]
            for i in range(0, len(trg)-1):
                out_word, hidd = dec(prev_word, hidd)
                prev_word = torch.argmax(out_word, dim=-1).detach()
                out[i] = out_word.view(-1)

        out = out.permute(1, 2, 0)
        trg = trg[1:].permute(1, 0)

        return out, trg


class LSTM_RNN_VLS_Learner(Learner):
    def __init__(self, modules, optimizers, loss_fn, models_directory):
        self.modules = modules
        self.optimizers = optimizers
        self.original_loss_fn = loss_fn
        self.models_directory = models_directory

    def loss_fn(self, out, trg):
        loss = 0
        for o, t in zip(out, trg):
            loss += self.original_loss_fn(o, t)
        return loss/len(out)

    def get_output_target(self, eval, example, src_name, trg_name,
                          trg_vocab_size, teacher_forcing, pad_src, pad_trg):
        enc, dec = self.modules
        src_raw, trg_raw = getattr(example, src_name), getattr(example, trg_name)

        src, trg, ls = pad_len_sort_both(src_raw, trg_raw, pad_src)
        hidd = enc(src, ls)

        if eval: teacher_forcing = 0
        use_teacher_forcing = True if random.random() < teacher_forcing else False

        outputs = []
        targets = []

        for i in range(trg.size(1)):
            seq = trg[:, i]
            inp = seq[(seq!=pad_trg).nonzero()]

            h = hidd[0][:, i][:, None, :]
            c = hidd[1][:, i][:, None, :]

            if use_teacher_forcing:
                out, _ = dec(inp[:-1], (h, c))

            else:
                out = torch.zeros(inp.size(0)-1, 1, trg_vocab_size).cuda()
                h = (h, c)
                prev_word = inp[0][None, :]
                for j in range(inp.size(0)-1):
                    out_word, h = dec(prev_word, h)
                    prev_word = torch.argmax(out_word, dim=-1).detach()
                    out[j] = out_word.view(-1)

            out = out.permute(1, 2, 0)
            inp = inp[1:].permute(1, 0)

            outputs.append(out)
            targets.append(inp)

        return outputs, targets
