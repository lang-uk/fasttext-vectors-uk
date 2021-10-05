from gensim.models.fasttext import FastText
from callbacks import EpochLogger, EpochSaver
import sys


def train_model(corpus_file, vec_size, algorithm, num_epoch, path_to_save_model, save_nth_epoch, min_n, max_n):
    epoch_logger = EpochLogger()
    epoch_saver = EpochSaver(path_to_save_model, save_nth_epoch)
    sg = 1 if algorithm == "skipgram" else 0
    model = FastText(vector_size=vec_size)

    # build the vocabulary
    print("\nBuilding vocabulary...\n")
    model.build_vocab(corpus_file=corpus_file)
    total_words = model.corpus_total_words

    print("\nStarted training...\n")
    # train the model

    model.train(corpus_file=corpus_file, sg = sg, total_words=total_words, epochs=num_epoch, min_n=min_n, max_n=max_n, callbacks=[epoch_logger, epoch_saver])


if __name__ == "__main__":
    # arguments reading
    corpus_path = sys.argv[1]
    vectors_size = int(sys.argv[2])
    algo = sys.argv[3]
    n_epoch = int(sys.argv[4])
    path_save_model = sys.argv[5]
    save_nth_weights = int(sys.argv[6])
    min_ngram = int(sys.argv[7])
    max_ngram = int(sys.argv[8])
    train_model(corpus_path, vectors_size, algo, n_epoch, path_save_model, save_nth_weights, min_ngram, max_ngram)
