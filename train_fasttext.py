import sys
import pathlib
import argparse
import logging

from gensim.models.fasttext import FastText
from callbacks import EpochLogger, EpochSaver


logger = logging.getLogger("fasttext")


def train_model(settings):
    assert settings.min_ngram < settings.max_ngram, "Min length of ngram cannot be greater or equal to the max length"
    epoch_logger = EpochLogger()
    epoch_saver = EpochSaver(settings)
    sg = 1 if settings.algo == "skipgram" else 0
    model = FastText(vector_size=settings.vectors_size)

    # build the vocabulary
    logger.info("Building vocabulary...")
    model.build_vocab(corpus_file=str(settings.corpus_path))
    total_words = model.corpus_total_words

    logger.info("Started training...")
    # train the model

    model.train(
        corpus_file=str(settings.corpus_path),
        sg=sg,
        total_words=total_words,
        epochs=settings.n_epoch,
        min_n=settings.min_ngram,
        max_n=settings.max_ngram,
        callbacks=[epoch_logger, epoch_saver],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fasttext vectors using gensim.")
    parser.add_argument("corpus_path", type=pathlib.Path, help="Path to the corpus text file")
    parser.add_argument("--vectors_size", type=int, help="Dimensions of the vectors", default=300)
    parser.add_argument("--n_epoch", type=int, help="Number of epochs", default=15)
    parser.add_argument("--save_nth_epoch", type=int, help="Save every nth epoch", default=5)
    parser.add_argument("--min_ngram", type=int, help="Min ngram", default=3)
    parser.add_argument("--max_ngram", type=int, help="Max ngram", default=6)
    parser.add_argument("path_save_model", type=pathlib.Path, help="Path to save checkpoints")
    parser.add_argument("--algo", type=str, help="Skipgrams or CBOW", choices=("skipgram", "cbow"), default="skipgram")
    parser.add_argument("--verbosity", type=int, default=1, choices=(0, 1, 2, 3))
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s:%(levelname)s %(asctime)s %(message)s")
    gensim_logger = logging.getLogger("gensim")

    if args.verbosity == 0:
        logger.setLevel(level=logging.WARNING)
        gensim_logger.setLevel(logging.WARNING)
    elif args.verbosity == 1:
        logger.setLevel(level=logging.INFO)
        gensim_logger.setLevel(logging.WARNING)
    elif args.verbosity == 2:
        logger.setLevel(level=logging.DEBUG)
        gensim_logger.setLevel(logging.INFO)
    elif args.verbosity == 3:
        logger.setLevel(level=logging.DEBUG)
        gensim_logger.setLevel(logging.DEBUG)

    train_model(args)
