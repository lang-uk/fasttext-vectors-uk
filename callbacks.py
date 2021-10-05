from gensim.models.callbacks import CallbackAny2Vec
import gensim
import time


class EpochSaver(CallbackAny2Vec):
    '''
    Callback to save model after each epoch.
    '''

    def __init__(self, path_prefix, save_nth_epoch):
        self.path_prefix = path_prefix
        self.save_nth_epoch = save_nth_epoch
        self.epoch = 0

    def on_epoch_end(self, model):
        if self.epoch % self.save_nth_epoch == 0:
            gensim.models.fasttext.save_facebook_model(model, f"{self.path_prefix}_epoch{self.epoch}.bin")
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    '''
    Callback to log information about training
    '''
    
    def __init__(self):
        self.epoch = 0
        self.epoch_time_start = 0

    def on_epoch_begin(self, model):
        self.epoch_time_start = time.time()
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        print(f"Time for training on {self.epoch} epoch: {time.time() - self.epoch_time_start}")
        print(f"Epoch #{self.epoch} end")
        self.epoch += 1
