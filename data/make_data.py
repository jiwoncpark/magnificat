from __future__ import print_function
import numpy as np
import pandas as pd
from dataloader import Dataloader
import time
import os, sys
data_path = os.path.join(os.environ['DEEPQSODIR'], 'data')
sys.path.insert(0, data_path)

print("Began making training data. Reading in source tables...")
lens_source_f = os.path.join(data_path, 'lens_tvar_source_table.csv')
nonlens_source_f = os.path.join(data_path, 'nonlens_source_table.csv')
print("Done reading in source tables.")

start = time.time()

X_path = os.path.join(data_path, 'features')
y_path = os.path.join(data_path, 'labels')

dl = Dataloader(lens_source_path=lens_source_f,
                nonlens_source_path=nonlens_source_f,
                observation_cutoff=65000, onehot_filters=True, debug=False)

dl.source_to_data(features_path=X_path,
                  labels_path=y_path, return_data=False)

end = time.time()
print("Done making training data in %0.2f seconds." %(end - start))