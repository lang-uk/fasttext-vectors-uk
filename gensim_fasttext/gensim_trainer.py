import sys
import pathlib
import argparse
import logging
import multiprocessing

from gensim.models.fasttext import FastText
from callbacks import EpochLogger, EpochSaver


logger = logging.getLogger("fasttext")


def train_model(settings):
    min_ngram, max_ngram = map(int, settings.ngram.split("-"))

    epoch_logger = EpochLogger()
    epoch_saver = EpochSaver(settings)
    sg = 1 if settings.algo == "skipgram" else 0
    weighted = 1 if settings.algo == "cbow_weighted" else 0
    model = FastText(
        vector_size=settings.vector_size,
        workers=settings.threads,
        position_dependent_weights=weighted,

        # Probably those hyperparamets contributed to initial bad performance of the GS
        window=15,
        negative=15,

        # Just to aligh all the parameters, but that doesn't make much sense, since:
        # In Facebook's FastText, "max length of word ngram" - but gensim only supports the
        # default of 1 (regular unigram word handling).
        # word_ngrams=3,
    )

    # build the vocabulary
    # TODO: try to move it outside of the grid loop
    logger.info("Building vocabulary...")
    model.build_vocab(corpus_file=str(settings.corpus_path))
    total_words = model.corpus_total_words

    logger.info(
        f"Started training (algo:{settings.algo}, dims:{settings.vector_size}, ngram:{settings.ngram}, corpus:{settings.corpus_path}, words:{total_words}) on {settings.threads} threads..."
    )
    # train the model

    model.train(
        corpus_file=str(settings.corpus_path),
        sg=sg,
        total_words=total_words,
        total_examples=model.corpus_count,
        epochs=settings.n_epoch,
        min_n=min_ngram,
        max_n=max_ngram,
        callbacks=[epoch_logger, epoch_saver],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train fasttext vectors using gensim. You can supply more than one value to some arguments to enable grid training"
    )
    parser.add_argument("corpus_path", type=pathlib.Path, help="Path to the corpus text file")
    parser.add_argument("--vector_size", type=int, help="Dimensions of the vectors", default=[300], nargs="+")
    parser.add_argument("--n_epoch", type=int, help="Number of epochs", default=10)
    parser.add_argument("--save_nth_epoch", type=int, help="Save every nth epoch", default=5)
    parser.add_argument(
        "--ngram",
        type=str,
        help="{min_n-max_n} ngram setting. Set max_n to be lesser than min_n to avoid char ngrams being used",
        default=["3-6"],
        nargs="+",
    )
    parser.add_argument("path_save_model", type=pathlib.Path, help="Path to save checkpoints")
    parser.add_argument(
        "--algo",
        type=str,
        help="Skipgrams or CBOW",
        choices=("skipgram", "cbow", "cbow_weighted"),
        default=["skipgram"],
        nargs="+",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help="Level of verbosity (3 means debug from everything, including gensim)",
    )
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
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

    logger.info(f"Training {len(args.ngram) * len(args.vector_size) * len(args.algo)} variants of word vectors")

    base_args = vars(args).copy()

    for algo in args.algo:
        for vector_size in args.vector_size:
            for ngram in args.ngram:
                base_args["algo"] = algo
                base_args["vector_size"] = vector_size
                base_args["ngram"] = ngram

                train_model(argparse.Namespace(**base_args))
