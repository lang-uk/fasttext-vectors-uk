# fasttext-vectors-uk

This repository contains a script to train word representations learned using the method described in [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606), aka FastText.
We use [gensim](https://radimrehurek.com/gensim/models/fasttext.html) library for training fasttext model. You can supply more than one value to some arguments to enable grid training

## Usage
```
git clone https://github.com/romanyshyn-natalia/fasttext-vectors-uk.git
cd fasttext-vectors-uk
pip install -r requirements.txt
python train_fasttext.py <corpus_file> <vector_size> <cbow/skipgram> <number_of_epoch> <output_file_path> <nth_epoch_to_save> <min_ngram> <max_ngram>
```

## Input parameters
```
corpus_path                   # path to the corpus text file
vector_size                   # dimensions of the vectors [300]
n_epoch                       # number of epochs [10]
save_nth_epoch                # save every nth epoch [5]
ngram                         # {min_n-max_n} ngram setting. Set max_n to be lesser than min_n to avoid char ngrams being used [3-6]
path_save_model               # path to save checkpoints
algo                          # skipgrams or CBOW [skipgram]
verbosity                     # level of verbosity (3 means debug from everything, including gensim) [1]
threads                       # number of threads [number of cpus]
```
