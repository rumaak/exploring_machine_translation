def pad_len_sort(t,pad):
    n_pad = (t==pad).sum(0).cpu().numpy()
    ls = t.size(0) - n_pad
    ids = n_pad.argsort()
    return t[:,ids], ls[ids]

def pad_len_sort_both(t1,t2,pad):
    n_pad = (t1==pad).sum(0).cpu().numpy()
    ls = t1.size(0) - n_pad
    ids = n_pad.argsort()
    return t1[:,ids], t2[:,ids], ls[ids]