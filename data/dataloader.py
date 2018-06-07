from __future__ import print_function
import numpy as np
import pandas as pd
from itertools import product
import gc
import os, sys
data_path = os.path.join(os.environ['DEEPQSODIR'], 'data')
sys.path.insert(0, data_path)
from data_utils import *

class Dataloader(object):
    
    """
    Class equipped with functions for generating the 
    datasets to be fed into the DNN from the input
    source tables.
    """
    
    def __init__(self, lens_source_path, nonlens_source_path, observation_cutoff=60150):
        self.lens_source_path = lens_source_path
        self.nonlens_source_path = nonlens_source_path
        self.lens = pd.read_csv(lens_source_path)
        self.nonlens = pd.read_csv(nonlens_source_path)
        
        self.NUM_TIMES = None # undefined until set_balance is called
        self.NUM_POSITIVES = self.lens['objectId'].nunique()
        self.NUM_FILTERS = 5
        self.seed = 123
        
        self.attributes = ['psf_fwhm', 'x', 'y', 'apFlux', 'apFluxErr', 
                           'apMag', 'apMagErr', 'trace', 'e1', 'e2', 'e', 'phi', 'd_time']
        self.NUM_ATTRIBUTES = len(self.attributes)
        
        self.filtered_attributes = [f + '_' + a for a, f in list(product(self.attributes, 'ugriz'))]
        assert len(self.filtered_attributes) == self.NUM_FILTERS * self.NUM_ATTRIBUTES
        
        self.observation_cutoff = observation_cutoff
        
        
    def set_balance(self, lens, nonlens, observation_cutoff):
        nonlens.query('MJD < @observation_cutoff', inplace=True) # giving up trace < 5.12
        # & objectId < (@min_nonlensid + @NUM_POSITIVES)
        self.NUM_TIMES = nonlens['MJD'].nunique()
        assert nonlens['MJD'].nunique() == lens['MJD'].nunique()
        # Get same number of lenses as lens sample
        final_nonlenses = nonlens['objectId'].unique()[: self.NUM_POSITIVES]
        nonlens = nonlens[nonlens['objectId'].isin(final_nonlenses)]
        gc.collect() 

        assert np.array_equal(lens.shape, nonlens.shape)
        
        return lens, nonlens
    
    def set_additional_columns(self, lens, nonlens):
        for src in [lens, nonlens]:
            # Add e, phi columns
            src['e'], src['phi'] = e1e2_to_ephi(e1=src['e1'], e2=src['e2'])
            # Set MJD relative to zero
            src['MJD'] = src['MJD'] - np.min(src['MJD'].values)
            # Map ccdVisitId to integers starting from 0
            sorted_obs_id = np.sort(src['ccdVisitId'].unique())
            time_map = dict(zip(sorted_obs_id, range(self.NUM_TIMES)))
            src['time_index'] = src['ccdVisitId'].map(time_map).astype(int)
            # Add a column of time elapsed since last observation, d_time
            src.sort_values(['objectId', 'MJD'], axis=0, inplace=True)
            src['d_time'] = src['MJD'] - src['MJD'].shift(+1)
            src['d_time'].fillna(0.0, inplace=True)
            src['d_time'] = np.clip(src['d_time'], a_min=0.0, a_max=None)
            src.drop(['MJD', 'ccdVisitId'], axis=1, inplace=True)
        gc.collect()
        
        return lens, nonlens
    
    def make_data_array(self, src, truth_value=1):
        
        print("NUM_TIMES: ", self.NUM_TIMES)
        print(src.shape[0])
        assert src.shape[0] == self.NUM_POSITIVES*self.NUM_TIMES
        
        # Pivot to get filters in each row
        src = src.pivot_table(index=['objectId', 'time_index'], 
                              columns=['filter'], 
                              values=self.attributes,
                              dropna=False)

        # Collapse multi-indexed column using filter_property formatting
        src.columns = src.columns.map('{0[1]}_{0[0]}'.format)

        assert np.array_equal(src.shape, 
                              (self.NUM_POSITIVES*self.NUM_TIMES, 
                               self.NUM_ATTRIBUTES*self.NUM_FILTERS))
        
        src.reset_index(inplace=True) #.set_index('objectId')
        gc.collect()

        assert np.array_equal(src.shape, 
                              (self.NUM_POSITIVES*self.NUM_TIMES, 
                               self.NUM_ATTRIBUTES*self.NUM_FILTERS + 2))
        assert np.array_equal(np.setdiff1d(src.columns.values, self.filtered_attributes), 
                              np.array(['objectId', 'time_index']))
        
        # Pivot to get time sequence in each row
        src = src.pivot_table(index=['objectId'], 
                            columns=['time_index'], 
                            values=self.filtered_attributes,
                            dropna=False)
        gc.collect()
        
        # Collapse multi-indexed column using time-filter_property formatting
        src.columns = src.columns.map('{0[1]}-{0[0]}'.format)
        #src = src.reindex(sorted(src.columns), axis=1, copy=False)
        
        self.timed_filtered_attributes = [str(t) + '-' + a for a, t in\
                                          list(product(self.filtered_attributes, range(self.NUM_TIMES)))]
        assert len(self.timed_filtered_attributes) == self.NUM_FILTERS*self.NUM_ATTRIBUTES*self.NUM_TIMES
        #assert np.array_equal(src.columns.values, np.sort(self.timed_filtered_attributes))
        assert np.array_equal(src.shape, 
                              (self.NUM_POSITIVES, 
                               self.NUM_ATTRIBUTES*self.NUM_FILTERS*self.NUM_TIMES))
        
        # Set null values to some arbitrary out-of-range value
        # TODO make the value even more meaningless
        src[src.isnull()] = -9999.0
        gc.collect()
        
        X = src.values.reshape(self.NUM_POSITIVES, 
                               self.NUM_FILTERS*self.NUM_ATTRIBUTES,
                               self.NUM_TIMES).swapaxes(1, 2)
        y = np.ones((self.NUM_POSITIVES, ))*truth_value
        
        return X, y
    
    def combine_lens_nonlens(self, lens_data, nonlens_data):
        X_lens, y_lens = lens_data
        X_nonlens, y_nonlens = nonlens_data
        
        np.random.seed(self.seed)
        
        assert np.array_equal(X_lens.shape, X_nonlens.shape)
        assert len(y_lens) == len(y_nonlens)
        
        X = np.concatenate([X_lens, X_nonlens], axis=0)
        y = np.concatenate([y_lens, y_nonlens], axis=0)
        
        return X, y
    
    def shuffle_data(self, X, y):
        
        p = np.random.permutation(self.NUM_POSITIVES)
        X = X[p, :]
        y = y[p, ]
        
        return X, y
    
    def source_to_data(self, features_path, label_path, return_data=False):
        import time
        
        start = time.time()
        
        self.lens, self.nonlens = self.set_balance(self.lens, 
                                                   self.nonlens, 
                                                   observation_cutoff=self.observation_cutoff)
        self.lens, self.nonlens = self.set_additional_columns(self.lens, self.nonlens)
        X_lens, y_lens = self.make_data_array(self.lens, truth_value=1)
        X_nonlens, y_nonlens = self.make_data_array(self.nonlens, truth_value=0)
        X, y = self.combine_lens_nonlens(lens_data=(X_lens, y_lens),
                                    nonlens_data=(X_nonlens, y_nonlens))
        X, y = self.shuffle_data(X, y)
        gc.collect()
        
        # Since savetxt only takes 1d or 2d arrays
        X = X.reshape(self.NUM_POSITIVES, self.NUM_FILTERS*self.NUM_ATTRIBUTES*self.NUM_TIMES)
        np.savetxt(features_path, X, delimiter=",")
        np.savetxt(label_path, y, delimiter=",")
        
        end = time.time()
        print("Done making the dataset in %0.2f seconds." %(end-start))
        
        if return_data:
            return X, y