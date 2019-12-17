import os
#Comment this out to use your gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras

import numpy
import pandas as pd
#import argparse

numpy.random.seed( 0 )

########
# PLAN #
########
'''
Create a neural network that takes a sequence of numbers and returns the next number

Very simple sanity check
'''

################
# CREATE MODEL #
################

max_features = 10

# Layers
model = Sequential()
model.add( Embedding( max_features, 128 ) )
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation=None))

#https://keras.io/examples/imdb_lstm/

metrics_to_output=[ 'binary_accuracy' ]
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )
model.summary()



#############
# CALLBACKS #
#############

csv_logger = tensorflow.keras.callbacks.CSVLogger( "training_log.csv", separator=',', append=False )
# Many fun options: https://keras.io/callbacks/
def schedule( epoch, lr ):
    if lr < 0.0001:
        return lr * 2
    return lr * 0.9
lrs = tensorflow.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

chkpt = tensorflow.keras.callbacks.ModelCheckpoint("checkpoint.{epoch:02d}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


callbacks=[csv_logger,lrs,chkpt]

#########
# TRAIN #
#########

class_weight = {0: 1.,
                1: 200.}

model.fit( x=input, y=output, batch_size=64, epochs=10, verbose=1, callbacks=callbacks, validation_data=(test_input,test_output), shuffle=True, class_weight=class_weight )


#############
# SPIN DOWN #
#############

model.save( args.model )
