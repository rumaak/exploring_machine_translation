import torch
import random

from src.utils import pad_len_sort_both

def Multi30k(enc,dec,train,valid,epochs,opt_enc,opt_dec,loss_fn, trg_vocab_size, trg_sos_id,
             teacher_forcing=0.5, end_train=300, end_val=100, print_every=100):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    def getLoss(val=False):
        data = train if not val else valid
        enc.eval()
        dec.eval() 
        avg_loss = 0
        example = 1
        for pair in data:
            src,trg = pair.de,pair.en
            hidd = enc(src)
            out = torch.zeros(trg.size(0)-1,trg.size(1),trg_vocab_size).cuda()
            
            prev_word = sos_token_tensor
            for i in range(0,len(trg)-1):
                out_word,hidd = dec(prev_word,hidd)
                prev_word = torch.argmax(out_word,dim=-1).detach()
                out[i] = out_word.view(-1)
            
            out = out.permute(1,2,0)
            trg = trg[1:].permute(1,0)
            
            loss = loss_fn(out,trg)
            avg_loss += loss.item()
            
            if example % end_val == 0:
                enc.train()
                dec.train()
                return avg_loss/example
            
            example += 1
    
    for e in range(1,epochs+1):
        example = 1
        avg_loss = 0
        checkpoint = 1
        for pair in train:
            src,trg = pair.de,pair.en
            hidd = enc(src)
            out = torch.zeros(trg.size(0)-1,trg.size(1),trg_vocab_size).cuda()
            
            opt_enc.zero_grad()
            opt_dec.zero_grad()
            
            use_teacher_forcing = True if random.random() < teacher_forcing else False
            
            if use_teacher_forcing:
                out,_ = dec(trg[:-1],hidd)
                
            else:
                prev_word = sos_token_tensor
                for i in range(0,len(trg)-1):
                    out_word,hidd = dec(prev_word,hidd)
                    prev_word = torch.argmax(out_word,dim=-1).detach()
                    out[i] = out_word.view(-1)
            
            out = out.permute(1,2,0)
            trg = trg[1:].permute(1,0)
            
            loss = loss_fn(out,trg)
            avg_loss += loss.item()
            loss.backward()
            
            opt_enc.step()
            opt_dec.step()
            
            if example % int(end_train/100) == 0:
                print(f'{round(example/(end_train/100),1)}% done')
            
            if example % print_every == 0:
                print(f'Train: {getLoss()} \nValid: {getLoss(True)} \n')
                avg_loss = 0
                torch.save(enc.state_dict(), f'models/LSTM_RNN/Multi30k/enc_{checkpoint}.pt')
                torch.save(dec.state_dict(), f'models/LSTM_RNN/Multi30k/dec_{checkpoint}.pt')
                checkpoint += 1
                
            if example % end_train == 0:
                break
                
            example += 1

def Multi30k_VLS(enc,dec,train,valid,epochs,opt_enc,opt_dec,loss_fn, trg_vocab_size, trg_sos_id, pad_de, pad_en,
                 teacher_forcing=0.5, end_train=300, end_val=100, print_every=100):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    def getLoss(val=False):
        data = train if not val else valid
        enc.eval()
        dec.eval() 
        avg_loss = 0
        example,batch_n = 1,1
        for batch in data:
            src_raw,trg_raw = batch.de,batch.en
            
            src,trg,ls = pad_len_sort_both(src_raw,trg_raw,pad_de)
            hidd = enc(src,ls)
            
            for i in range(trg.size(1)):
                seq = trg[:,i] 
                inp = seq[(seq!=pad_en).nonzero()]

                h = hidd[0][:,i][:,None,:]
                c = hidd[1][:,i][:,None,:]

                out = torch.zeros(inp.size(0)-1,1,trg_vocab_size).cuda()
                h = (h,c)
                prev_word = sos_token_tensor
                for j in range(inp.size(0)-1):                    
                    out_word,h = dec(prev_word,h)
                    prev_word = torch.argmax(out_word,dim=-1).detach()
                    out[j] = out_word.view(-1) # FLAG

                out = out.permute(1,2,0)
                inp = inp[1:].permute(1,0)

                loss = loss_fn(out,inp)
                avg_loss += loss.item()
                
                example += 1
            
            if batch_n % end_val == 0:
                enc.train()
                dec.train()
                return avg_loss/(batch_n*example)
        
            batch_n += 1
            example = 1
    
    for e in range(1,epochs+1):
        batch_n = 1
        batch_loss = 0
        checkpoint = 1
        for batch in train:
            src_raw,trg_raw = batch.de,batch.en
            
            src,trg,ls = pad_len_sort_both(src_raw,trg_raw,pad_de)
            hidd = enc(src,ls)
            
            opt_enc.zero_grad()
            opt_dec.zero_grad()
            
            use_teacher_forcing = True if random.random() < teacher_forcing else False
            
            if use_teacher_forcing:
                for i in range(trg.size(1)):
                    seq = trg[:,i] 
                    inp = seq[(seq!=pad_en).nonzero()]
                    
                    h = hidd[0][:,i][:,None,:]
                    c = hidd[1][:,i][:,None,:]
                    
                    out,_ = dec(inp[:-1],(h,c))
                    
                    out = out.permute(1,2,0)
                    inp = inp[1:].permute(1,0)

                    loss = loss_fn(out,inp)
                    batch_loss += loss
                    
            else:
                for i in range(trg.size(1)):
                    seq = trg[:,i] 
                    inp = seq[(seq!=pad_en).nonzero()]
                    
                    h = hidd[0][:,i][:,None,:]
                    c = hidd[1][:,i][:,None,:]
                    
                    out = torch.zeros(inp.size(0)-1,1,trg_vocab_size).cuda()
                    h = (h,c)
                    prev_word = sos_token_tensor
                    for j in range(inp.size(0)-1):                    
                        out_word,h = dec(prev_word,h)
                        prev_word = torch.argmax(out_word,dim=-1).detach()
                        out[j] = out_word.view(-1) # FLAG
                    
                    out = out.permute(1,2,0)
                    inp = inp[1:].permute(1,0)

                    loss = loss_fn(out,inp)
                    batch_loss += loss
            
            batch_loss.backward()
            
            opt_enc.step()
            opt_dec.step()
            
            if batch_n % int(end_train/100) == 0:
                print(f'{round(batch_n/(end_train/100),1)}% done')
            
            if batch_n % print_every == 0:
                print(f'Train: {getLoss()} \nValid: {getLoss(True)} \n')
                torch.save(enc.state_dict(), f'models/LSTM_RNN/Multi30k/enc_{checkpoint}.pt')
                torch.save(dec.state_dict(), f'models/LSTM_RNN/Multi30k/dec_{checkpoint}.pt')
                checkpoint += 1
                
            if batch_n % end_train == 0:
                break
                
            batch_n += 1
            batch_loss = 0

def WMT14(enc,dec,train,valid,epochs,opt_enc,opt_dec,loss_fn, trg_vocab_size, trg_sos_id,
          teacher_forcing=0.5, end=300, print_every=100):
    sos_token_tensor = torch.LongTensor([[trg_sos_id]]).cuda()
    num_valid = int(end/100)
    def getLoss(val=False):
        data = train if not val else valid
        enc.eval()
        dec.eval() 
        avg_loss = 0
        example = 1
        for pair in data:
            src,trg = pair.src,pair.trg
            hidd = enc(src)
            out = torch.zeros(trg.size(0)-1,trg.size(1),trg_vocab_size).cuda()
            
            prev_word = sos_token_tensor
            for i in range(0,len(trg)-1):
                out_word,hidd = dec(prev_word,hidd)
                prev_word = torch.argmax(out_word,dim=-1).detach()
                out[i] = out_word.view(-1)
            
            out = out.permute(1,2,0)
            trg = trg[1:].permute(1,0)
            
            loss = loss_fn(out,trg)
            avg_loss += loss.item()
            
            if example % num_valid == 0:
                enc.train()
                dec.train()
                return avg_loss/example
            
            example += 1
    
    for e in range(1,epochs+1):
        example = 1
        avg_loss = 0
        for pair in train:
            src,trg = pair.src,pair.trg
            hidd = enc(src)
            out = torch.zeros(trg.size(0)-1,trg.size(1),trg_vocab_size).cuda()
            
            opt_enc.zero_grad()
            opt_dec.zero_grad()
            
            use_teacher_forcing = True if random.random() < teacher_forcing else False
            
            if use_teacher_forcing:
                out,_ = dec(trg[:-1],hidd)
                
            else:
                prev_word = sos_token_tensor
                for i in range(0,len(trg)-1):
                    out_word,hidd = dec(prev_word,hidd)
                    prev_word = torch.argmax(out_word,dim=-1).detach()
                    out[i] = out_word.view(-1)
            
            out = out.permute(1,2,0)
            trg = trg[1:].permute(1,0)
            
            loss = loss_fn(out,trg)
            avg_loss += loss.item()
            loss.backward()
            
            opt_enc.step()
            opt_dec.step()
            
            if example % int(end/100) == 0:
                print(f'{round(example/(end/100),1)}% done')
            
            if example % print_every == 0:
                print(f'Train: {getLoss()} \nValid: {getLoss(True)} \n')
                avg_loss = 0
                
            if example % end == 0:
                break
                
            example += 1