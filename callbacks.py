import os.path
import time
import logging

import gensim
from gensim.models.callbacks import CallbackAny2Vec

logger = logging.getLogger("fasttext")


class EpochSaver(CallbackAny2Vec):
    """
    Callback to save model after each epoch.
    """

    def __init__(self, settings):
        self.settings = settings
        self.epoch = 0

    def on_epoch_end(self, model):
        if (self.epoch + 1) % self.settings.save_nth_epoch == 0:
            gensim.models.fasttext.save_facebook_model(
                model,
                str(self.settings.path_save_model
                / f"{os.path.basename(self.settings.corpus_path)}.d{self.settings.vectors_size}.subword{self.settings.min_ngram}-{self.settings.max_ngram}.{self.settings.algo}.epoch{self.epoch}.bin"),
            )

        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    """
    Callback to log information about training
    """

    def __init__(self):
        self.epoch = 0
        self.epoch_time_start = 0

    def on_epoch_begin(self, model):
        self.epoch_time_start = time.time()
        logger.info(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        logger.info(f"Time for training on {self.epoch} epoch: {time.time() - self.epoch_time_start}")
        logger.info(f"Epoch #{self.epoch} end")
        self.epoch += 1
