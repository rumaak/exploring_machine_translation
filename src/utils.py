import matplotlib.pyplot as plt


def pad_len_sort(t, pad):
    n_pad = (t == pad).sum(0).cpu().numpy()
    ls = t.size(0) - n_pad
    ids = n_pad.argsort()
    return t[:, ids], ls[ids]


def pad_len_sort_both(t1, t2, pad):
    n_pad = (t1 == pad).sum(0).cpu().numpy()
    ls = t1.size(0) - n_pad
    ids = n_pad.argsort()
    return t1[:, ids], t2[:, ids], ls[ids]


def plotAttention(attention, source, target):
    plt.figure(figsize=(18, 18))
    plt.imshow(attention, interpolation='nearest', cmap='gray', aspect='auto')
    ax = plt.gca()

    ax.locator_params(nbins=len(target), axis='x')
    ax.set_xticklabels(['0'] + target)

    ax.locator_params(nbins=len(source), axis='y')
    ax.set_yticklabels(['0'] + source)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
