# Neural machine translation
In this project I have attempted to implement neural machine translation algorithm(s) together with bunch of other complementary stuff -
data preprocessing, etc. The **Machine translation.ipynb** notebook puts together all the machinery and presents the work done. Before
proceeding, see **Prerequisities** below.

History of this repo is quite big, so it is wise to clone only the last commit. In future, I am planning to delete this repo and create a fresh new one without this issue.

### Prerequisities
##### Datasets
* WMT14  
  * download https://1drv.ms/f/s!AiQ5a2cXVytTlkZD0HQn5FdGRgB2  
  * move to data/WMT14

* Multi30k  
  * download https://1drv.ms/f/s!AiQ5a2cXVytTllU4usk93QbPGv1s  
  * move to data/Multi30k

##### Environment
I highly recommend setting up a conda environment before installing packages.
* navigate to root
* pip install -r requirements.txt

Also, while in environment, do this:
`python -m spacy download en`

##### Trained models
* LSTM RNN
   * download https://1drv.ms/f/s!AiQ5a2cXVytTmUHOdj_FIim0ge2P
   * move to models/LSTM_RNN

* RNNsearch
   * download https://1drv.ms/f/s!AiQ5a2cXVytTmUTIIBY09Kq4NSC4
   * move to models/RNNsearch

