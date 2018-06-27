from trainer import Trainer
from keras.optimizers import Adam
from keras.callbacks import (Callback, TensorBoard, EarlyStopping,
                             ModelCheckpoint, CSVLogger, ProgbarLogger)
'''
if 'get_ipython' in vars() and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from keras_tqdm import TQDMNotebookCallback as Progbar
else:
    from keras_tqdm import TQDMCallback
    import sys
    class Progbar(TQDMCallback):  # redirect TQDMCallback to stdout
        def __init__(self):
            TQDMCallback.__init__(self)
            self.output_file = sys.stdout
'''

class Trainer_autoencoder(Trainer):
    
    def __init__(self, model, pretrained_weights=False, train=True):
        Trainer.__init__(self, model, pretrained_weights, train)
        #super(Autoencoder_trainer, self).__init__(model, pretrained_weights, train)
        pass
    
    def get_run_id(self):
        pass
    
    def save_checkpoint(self):
        pass
    
    def log_metrics(self):
        pass
 
    def run_training(self, X, y, train_config=None):
        optimizer = Adam(lr=1.e-5)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['binary_accuracy'],
                      sample_weight_mode=None)
        history = self.model.fit(x=X, y=y, epochs=50, batch_size=500, sample_weight=None,
                          callbacks=[ProgbarLogger,
                                     CSVLogger(self.csvlog_path, separator=',', append=False),
                                     TensorBoard(log_dir=self.log_dir, write_graph=False),
                                     ModelCheckpoint(self.weights_dir,
                                                     save_weights_only=True),],
                          verbose=False,
                          validation_split=0.2,
                          shuffle=True)
        return history

    
    