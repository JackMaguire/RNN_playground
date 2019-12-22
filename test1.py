import os
#Comment this out to use your gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras

import numpy as np
import pandas as pd
#import argparse

np.random.seed( 0 )

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
#model.add( LSTM( 10, input_shape=( None, 5, 1 ) ) )
model.add( LSTM( 10, input_shape=( 5, 1 ) ) )
model.add( Dense( 3, activation='sigmoid' ) )

#https://keras.io/examples/imdb_lstm/

metrics_to_output=[ 'binary_accuracy' ]
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )
#model.summary()

train_input = []
train_output = []

# Output is one of 3 classes: increasing, up-down, or decreasing

# Input is >0, -1.0 denotes blank space

train_input.append( [ [ 0.0 ], [ 0.5 ], [ 1.0 ], [ -1.0 ], [ -1.0 ] ] )
train_output.append( [ 0, 0, 1 ] )

train_input.append( [ [ 0.0 ], [ 2.0 ], [ 0.5 ], [ -1.0 ], [ -1.0 ] ] )
train_output.append( [ 0, 1, 0 ] )

train_input.append( [ [ 0.0 ], [ 0.5 ], [ 1.0 ], [ 5.0 ], [ 10.0 ] ] )
train_output.append( [ 0, 0, 1 ] )

train_input.append( [ [ 5.0 ], [ 2.0 ], [ -1.0 ], [ -1.0 ], [ -1.0 ] ] )
train_output.append( [ 1, 0, 0 ] )

train_input.append( [ [ 5.0 ], [ 2.0 ], [ 1.0 ], [ 0.5 ], [ 0.0 ] ] )
train_output.append( [ 1, 0, 0 ] )

train_input.append( [ [ 10.0 ], [ 2.0 ], [ 1.0 ], [ 0.5 ], [ 0.0 ] ] )
train_output.append( [ 1, 0, 0 ] )

my_input = np.asarray( train_input )
my_output = np.asarray( train_output )

model.fit( x=my_input, y=my_output, batch_size=1, epochs=10 )
