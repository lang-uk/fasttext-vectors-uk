# Learning Ukrainian word vectors with fastText models

`run_the_grid.py` is the script that allows a grid training of fasttext on a given corpus on different servers (nodes). It uses google spreadsheet (through gspread) to perform task distribution and to write down the results. This repo also contains the previous version of the script (`gensim_fasttext`), that was based on gensim. However, on our data, vectors trained on gensim shown weaker performance on [our intrinsic tests](https://github.com/lang-uk/vecs), so we left it in the repo for the reference.

Our rationale was to train as much as 96 different combinations of hyperparams on the same corpus using farm of 7 servers (estimated running time is 25-30 days). Script do not upload the results of training, we've used https://github.com/prasmussen/gdrive to upload files manually from the workers. 

## Installation
To install from our github repository, you can do:
```bash
git clone https://github.com/romanyshyn-natalia/fasttext-vectors-uk.git
cd fasttext-vectors-uk
```

## Requirements
The following command installs all necessary packages:
```bash
pip install -r requirements.txt
```

## Usage
Script has two commands, `setup` and `train`. First allows you to create folders, download and unpack vectors, checkout and compile Facebook's fasttext binaries and store the config. You'll need to create your own spreadsheet (you can copy it from [here](https://docs.google.com/spreadsheets/d/150DjEZKCuJEcsCJWahWmhPkfHzn9pA-N3UIYYx7XM04/edit?usp=sharing)) to manage tasks distribution. Spreadsheet contains two worksheets, first has the hyperparams you'd like to try in the grid and stats, second is generated from the hyperparams (first two columns) and is being updated by worker nodes (columns 3-5). The last column is updated manually after the upload of the vectors. You'll also need to create a json file for gspread with access credentials and store in the `api_keys` folder. You would probably like to replace the hostname with `--hostname` and adjust number of threads (`--threads`, defaulted to the number of CPU cores -2).

You should also specify the id of your google spreadsheet with `--spreadshet_id` (yes, we are aware of a typo there) and specify your api key file location with `--api_key_location`. Please refer to the original documentation of the gspread for a [perfect step-by-step guide](https://docs.gspread.org/en/latest/oauth2.html#enable-api-access-for-a-project).

```bash
python ./run_the_grid.py -v setup
```

## Parameters
```bash

~/fasttext-vectors-uk$ python run_the_grid.py setup --help
usage: run_the_grid.py setup [-h] [--overwrite_config] [--overwrite_corpus] [--overwrite_fasttext]
                             [--corpus_location CORPUS_LOCATION] [--corpus_url CORPUS_URL]
                             [--fasttext_location FASTTEXT_LOCATION] [--vectors_location VECTORS_LOCATION]
                             [--api_key_location API_KEY_LOCATION] [--spreadshet_id SPREADSHET_ID] [--threads THREADS]
                             [--logfile LOGFILE] [--hostname HOSTNAME]

optional arguments:
  -h, --help            show this help message and exit
  --overwrite_config    Overwrite config if it is already exists
  --overwrite_corpus    Overwrite corpus if it is already exists
  --overwrite_fasttext  Download and rebuild fasttext, if it is already exists
  --corpus_location CORPUS_LOCATION
                        Download corpus to specific folder
  --corpus_url CORPUS_URL
                        Download corpus to specific folder
  --fasttext_location FASTTEXT_LOCATION
                        Download and build fasttext to specific folder
  --vectors_location VECTORS_LOCATION
                        Store vectors to given folder after the training
  --api_key_location API_KEY_LOCATION
                        Location of json file with service account credentials for google drive and spreadsheet
  --spreadshet_id SPREADSHET_ID
                        Google Spreadsheet id (the one from the url) with the spreadsheet of tasks and results
  --threads THREADS     Number of threads to use
  --logfile LOGFILE     JSONLines file to write training details
  --hostname HOSTNAME   Identifier of the worker, defaulted to the hostname
```
You might also modify generated `config.json` manually or re-run the `setup` command with the `--overwrite_config` key. 

Once you are done with the setup on your first node, you might run the training immediatelly on that machine.
To do so run 
```bash
python ./run_the_grid.py -v train
```

It'll run handful of preflight checks, connect to the spreadsheet, pick the next available task, set it to the `Processing` state and start training.

## Caveats
* Script is training vectors of dimensionality 300.
* Script deletes textual version of the vectors once the training is done. For our purposes we only need binary vectors.
* In the training mode script will be start training the next task as soon as the current one is finished (until there are no more tasks or the Universe came to it's end, or you ran out of space (see below)).
* You **should** watch for the free space on your servers. With vectors of dim 300 and enough textual data (our corpus has almost 20GB of texts) generated vector file weights 8.3GB each, so you might ran out of space quickly. It also needs some temporary space to store textual version of the vectors.
* Script is maintaining the local log file `log.jsonl` for your reference.
* It might take weeks, months and even years to train a huge grid. Throw as many servers as you can. Oracle Free Tier Ampere servers is a good working horse (12 ARM cores and 72GB for free!) and probably the best performance that we had on different arcitectures (old Intel Xeons, Intel i7, AMD etc). Unfortunatelly, it's impossible to train fasttext on GPU.
* Also, run the training in `tmux` or `screen`
* No, you cannot use google colab to train vectors, as they only has 2 CPU cores and 24 hours is not enough to finish the task (even with Google Colab Pro Plus).


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate (ha-ha).
