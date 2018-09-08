import torch

from src.utils import pad_len_sort

# TODO: Format string to torch.Tensor
# TODO: Give functions proper names

def Multi30k(enc,dec,trg_sos_id,trg_eos_id,sent):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    enc.eval()
    dec.eval()
    hidd = enc(sent)
    out = []
    prev_word = sos_token_tensor
    while prev_word != trg_eos_id and len(out) < 50:
        out_word,hidd = dec(prev_word,hidd)
        prev_word = torch.argmax(out_word,dim=-1)
        out.append(prev_word.item())
    return out

def Multi30k_VLS(enc,dec,trg_sos_id,trg_eos_id,src_pad_id,sent):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    enc.eval()
    dec.eval()
    src,ls = pad_len_sort(sent,src_pad_id)
    hidd = enc(src,ls)
    out = []
    prev_word = sos_token_tensor
    while prev_word != trg_eos_id and len(out) < 50:
        out_word,hidd = dec(prev_word,hidd)
        prev_word = torch.argmax(out_word,dim=-1)
        out.append(prev_word.item())
    return out

def WMT14(enc,dec,trg_sos_id,trg_eos_id,sent):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    enc.eval()
    dec.eval()
    hidd = enc(sent)
    out = []
    prev_word = sos_token_tensor
    while prev_word != trg_eos_id and len(out) < 50:
        out_word,hidd = dec(prev_word,hidd)
        prev_word = torch.argmax(out_word,dim=-1)
        out.append(prev_word.item())
    return out

def rnnsearch(enc,dec,trg_sos_id,trg_eos_id,sent):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    enc.eval()
    dec.eval()
    old_out,_ = enc(sent)
    out = []
    prev_word = sos_token_tensor
    state = None
    while prev_word != trg_eos_id and len(out) < 50:
        out_word,hidd = dec(old_out,state,prev_word)
        prev_word = torch.argmax(out_word,dim=-1).detach()
        out.append(prev_word.item())
    return out