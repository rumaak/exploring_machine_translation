import torch

from src.utils import pad_len_sort


def lstm_rnn(enc,dec,trg_sos_id,trg_eos_id,sent):
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

def lstm_rnn_vls(enc,dec,trg_sos_id,trg_eos_id,src_pad_id,sent):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    enc.eval()
    dec.eval()
    src,ls = pad_len_sort(sent,src_pad_id)
    hidd = enc(src,ls)
    out = []
    prev_word = sos_token_tensor
    while prev_word != trg_eos_id and len(out) < 30:
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
    att = []
    prev_word = sos_token_tensor
    state = None
    while prev_word != trg_eos_id and len(out) < 30:
        out_word,state,a = dec(old_out,state,prev_word,getAttention=True)
        prev_word = torch.argmax(out_word,dim=-1).detach()

        out.append(prev_word.item())
        att.append(a.detach().cpu())

    att = torch.cat(att, 1)[:,:,0].numpy()

    return out, att
