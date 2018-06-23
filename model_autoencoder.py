# Base framework provided by
# https://github.com/bnaul/IrregularTimeSeriesAutoencoderPaper

from keras.layers import (Input, Dense, LSTM, Dropout,
                          Flatten, RepeatVector, Bidirectional, TimeDistributed)
from keras.layers.merge import concatenate
from keras.models import Model

class Autoencoder(object):
    
    def __init__(self, encoder_config=None, decoder_config=None):
        pass
    
    def encode(self, model_input, output_size=8, num_units=64, num_layers=2, drop_frac=0.25, bidirectional=False):
        encoded = model_input
        if bidirectional:
                direction = Bidirectional
        else:
            direction = lambda x: x
        for l in range(num_layers):
            encoded = direction(LSTM(units=num_units, 
                                        name='encode_{}'.format(l),
                                        return_sequences=(l < num_layers - 1)))(encoded)
            if drop_frac > 0.0:
                encoded = Dropout(rate=drop_frac, 
                                 name='drop_encode_{}'.format(l))(encoded)
        encoded = Dense(units=output_size,
                       activation='linear',
                       name='fc_encode')(encoded)
        return encoded
    
    def decode(self, encoded, output_size=738, d_time=None, num_units=64, num_layers=2, drop_frac=0.25, bidirectional=False):
        decoded = RepeatVector(output_size, name='repeat')(encoded)
        if d_time is not None:
            decoded = concatenate([d_time, decoded])
        for l in range(num_layers):
            if drop_frac > 0.0 and l > 0:  # skip these for first layer for symmetry
                decoded = Dropout(rate=drop_frac, 
                                  name='drop_decode_{}'.format(l))(decoded)
            if bidirectional:
                direction = Bidirectional
            else:
                direction = lambda x: x
            decoded = direction(LSTM(units=output_size,
                                     name='decode_{}'.format(l),
                                     return_sequences=True))(decoded)

        decoded = TimeDistributed(Dense(units=1,
                                        activation='linear'),
                                  name='time_dist')(decoded)
        return decoded           
    
    def autoencode(self, model_input):
        encoded = self.encode(model_input=model_input)
        decoded = self.decode(encoded=encoded)
        autoencoded = Model(inputs=model_input, outputs=decoded)

        return autoencoded
    
if __name__== '__main__':
    autoencoder = Autoencoder()
    model_input = Input(shape=(738, 2), name='main_input')
    model = autoencoder.autoencode(model_input)
    print(model.summary())
    