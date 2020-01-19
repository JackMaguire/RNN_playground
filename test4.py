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
input = Input(shape=(5,1,), name="in1", dtype="float32" )
#exit( 0 )
lstm1 = LSTM( 10, input_shape=(5,1), return_sequences="true" )( input )
#exit( 0 )
lstm2 = LSTM( 5, return_sequences="false" )( lstm1 )
#exit( 0 )
flat = Flatten( )( lstm2 )

d1 = Dense( units=10, activation="relu" )( input )
d2 = Dense( units=10, activation="relu" )( d1 )
d3 = Dense( units=10, activation="relu" )( d2 )
flat2 = Flatten( )( d3 )

merge = tensorflow.keras.layers.concatenate( [flat,flat2], name="merge", axis=-1 )


out = Dense( units=3, activation="relu" )( merge )

model = Model( inputs=input, outputs=out )
metrics_to_output=[ 'binary_accuracy' ]
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )

model.summary()

train_input = []
train_output = []

# Output bits are: (unfinished), (has a 2.0), (strictly decreasing, ignoring -1.0s)

# Input is >0, -1.0 denotes blank space

train_input.append( [ [ 0.0 ], [ 0.5 ], [ 1.0 ], [ -1.0 ], [ -1.0 ] ] )
train_output.append( [ 1, 0, 0 ] )

train_input.append( [ [ 0.0 ], [ 2.0 ], [ 0.5 ], [ -1.0 ], [ -1.0 ] ] )
train_output.append( [ 1, 1, 0 ] )

train_input.append( [ [ 0.0 ], [ 0.5 ], [ 1.0 ], [ 5.0 ], [ 10.0 ] ] )
train_output.append( [ 0, 0, 0 ] )

train_input.append( [ [ 5.0 ], [ 2.0 ], [ -1.0 ], [ -1.0 ], [ -1.0 ] ] )
train_output.append( [ 1, 1, 1 ] )

train_input.append( [ [ 5.0 ], [ 2.0 ], [ 1.0 ], [ 0.5 ], [ 0.0 ] ] )
train_output.append( [ 0, 1, 1 ] )

train_input.append( [ [ 10.0 ], [ 2.0 ], [ 1.0 ], [ 0.5 ], [ 0.0 ] ] )
train_output.append( [ 0, 1, 1 ] )

my_input = np.asarray( train_input )
my_output = np.asarray( train_output )

model.fit( x=my_input, y=my_output, batch_size=6, epochs=100 )


predictions = model.predict( my_input )

for i in range( 0, len( my_output ) ):
    print( "Predicted ", predictions[i], " instead of ", my_output[ i ] )

model.summary()
