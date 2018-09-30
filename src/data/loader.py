import string
import re

import spacy

from torchtext.data import Field, TabularDataset, BucketIterator, interleave_keys

tok = spacy.load('en').tokenizer

def load_multi30k(bs):
	train_path = 'data/Multi30k/processed/train.en_de'
	val_path = 'data/Multi30k/processed/val.en_de'

	def tokenize(text):
	    text = text.translate(str.maketrans('', '', string.punctuation))
	    text = re.sub(r'\d+',r'num',text)
	    return [x.text for x in tok(text) if x.text != ' ']

	DE = Field(lower=True,tokenize=tokenize,eos_token='<EOS>')
	EN = Field(lower=True,tokenize=tokenize,eos_token='<EOS>',init_token='<SOS>')

	train_ds = TabularDataset(train_path,'tsv',[('en',EN),('de',DE)],skip_header=True)
	val_ds = TabularDataset(val_path,'tsv',[('en',EN),('de',DE)],skip_header=True)

	train_iter = BucketIterator(train_ds,
	                            bs,
	                            sort_key=lambda x: interleave_keys(len(x.en),len(x.de)),
	                            shuffle=True,
	                            sort_within_batch=True)
	valid_iter = BucketIterator(val_ds,
	                            bs,
	                            sort_key=lambda x: interleave_keys(len(x.en),len(x.de)),
	                            shuffle=True,
	                            sort_within_batch=True)

	DE.build_vocab(train_ds.de)
	EN.build_vocab(train_ds.en)

	return train_iter, valid_iter, DE, EN

def load_WMT14(bs):
	data_path = 'data/WMT14/processed/train.cs_en.sample.tsv'

	to_delete = [r'&quot;',r'&apos;']

	def tokenize(text):
	    for p in to_delete:
	        text = re.sub(p,'',text)
	    text = text.translate(str.maketrans('', '', string.punctuation))
	    text = re.sub(r'\d+',r'num',text)
	    return [x.text for x in tok(text) if x.text != ' ']

	CZ = Field(lower=True,tokenize=tokenize,eos_token='<EOS>')
	EN = Field(lower=True,tokenize=tokenize,eos_token='<EOS>',init_token='<SOS>')

	data_ds = TabularDataset(data_path,'tsv',[('src',CZ),('trg',EN)],skip_header=True)
	train,valid = data_ds.split(split_ratio=0.8)

	train_iter = BucketIterator(train,
	                            bs,
	                            sort_key=lambda x: interleave_keys(len(x.src),len(x.trg)),
	                            shuffle=True,
	                            sort_within_batch=True)
	valid_iter = BucketIterator(valid,
	                            bs,
	                            sort_key=lambda x: interleave_keys(len(x.src),len(x.trg)),
	                            shuffle=True,
	                            sort_within_batch=True)

	CZ.build_vocab(data_ds.src,min_freq=2)
	EN.build_vocab(data_ds.trg,min_freq=3)

	return train_iter, valid_iter, CZ, EN