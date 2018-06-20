from __future__ import print_function
import numpy as np
import pandas as pd
from dataloader import Dataloader
import os, sys
data_path = os.path.join(os.environ['DEEPQSODIR'], 'data')
sys.path.insert(0, data_path)

lens_source_f = os.path.join(data_path, 'lens_tvar_source_table.csv')
nonlens_source_f = os.path.join(data_path, 'nonlens_source_table.csv')

X_path = os.path.join(data_path, 'features.csv')
y_path = os.path.join(data_path, 'labels.csv')

dl = Dataloader(lens_source_path=lens_source_f,
                nonlens_source_path=nonlens_source_f,
                observation_cutoff=60000, onehot_filters=True, debug=True)

dl.source_to_data(features_path=X_path,
                  label_path=y_path, return_data=False)

