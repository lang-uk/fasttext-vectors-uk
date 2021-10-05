# fasttext-vectors-uk

This repository contains a script to train word representations learned using the method described in [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606), aka FastText.
We use [gensim](https://radimrehurek.com/gensim/models/fasttext.html) library for training fasttext model. 

## Usage
```
git clone https://github.com/romanyshyn-natalia/fasttext-vectors-uk.git
cd fasttext-vectors-uk
pip install -r requirements.txt
python train_fasttext.py <corpus_file> <vector_size> <cbow/skipgram> <number_of_epoch> <output_file_path> <nth_epoch_to_save> <min_ngram> <max_ngram>
```

## Input parameters
```
corpus_file,            # training file path
vector_size,            # size of word vectors
algorithm,              # unsupervised fasttext model {cbow, skipgram}
number_of_epoch,        # number of epochs to train
output_file_path,       # path to file where the word representations will be saved
save_nth_epoch,         # save n_th epoch
min_ngram               # min length of char ngram
max_ngram               # max length of char ngram
```
