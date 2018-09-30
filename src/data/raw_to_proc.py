import pandas as pd
import numpy as np

def proc_multi30k():
	train_en = 'data/Multi30k/raw/train.en'
	train_de = 'data/Multi30k/raw/train.de'

	val_en = 'data/Multi30k/raw/val.en'
	val_de = 'data/Multi30k/raw/val.de'

	train = pd.DataFrame()
	train['en'] = pd.read_table(train_en,header=None).iloc[:,0]
	train['de'] = pd.read_table(train_de,header=None).iloc[:,0]

	val = pd.DataFrame()
	val['en'] = pd.read_table(val_en,header=None).iloc[:,0]
	val['de'] = pd.read_table(val_de,header=None).iloc[:,0]

	train_path = 'data/Multi30k/processed/train.en_de'
	val_path = 'data/Multi30k/processed/val.en_de'

	train.to_csv(train_path,sep='\t',index=False)
	val.to_csv(val_path,sep='\t',index=False)

def proc_WMT14():
	data_en = 'data/WMT14/raw/train.en.tsv'
	data_cz = 'data/WMT14/raw/train.cs.tsv'

	data = pd.DataFrame()
	data['src'] = pd.read_table(data_cz,header=None).iloc[:,0]
	data['trg'] = pd.read_table(data_en,header=None).iloc[:,0]

	data_path = 'data/WMT14/processed/train.cs_en.tsv'

	data.to_csv(data_path,sep='\t',index=False)

def create_WMT14_samp(ratio=0.1):
	data = pd.read_table('data/WMT14/processed/train.cs_en.tsv',header=0)

	l = len(data)
	n_samp = int(l*ratio)
	ids = np.random.randint(0,l,size=n_samp)

	sample_df = data.iloc[ids]
	sample_df.to_csv('data/WMT14/processed/train.cs_en.sample.tsv',sep='\t',index=False)