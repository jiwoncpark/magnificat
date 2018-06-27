import os, sys
root_path = os.path.join(os.environ['DEEPQSODIR'])
sys.path.insert(0, root_path)


class Trainer(object):
    
    def __init__(self, model, pretrained_weights=False, train=True):
        self.train = train
        self.model = model
        self.pretrained_weights = pretrained_weights
        self.loaded = False
        
        self.log_dir = os.path.join(root_path, 'keras_log')
        self.weights_dir = os.path.join(self.log_dir, 'weights')
        self.csvlog_path = os.path.join(self.log_dir, 'training.csv')
        
        if os.path.exists(self.weights_dir) and pretrained_weights:
            print("Loading {}...".format(self.weights_dir))
            self.model.load_weights(self.weights_dir)
            self.loaded = True
        elif not self.train:
            # Can't run prediction without existing weights!
            raise FileNotFoundError("No weights found in {}.".format(self.log_dir))
        elif self.train:
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.weights_dir, exist_ok=True)
        
    
    def get_run_id(self):
        pass
    
    def save_checkpoint(self):
        pass
    
    def log_metrics(self):
        pass
 
    def run_training(self):
        pass
    
    