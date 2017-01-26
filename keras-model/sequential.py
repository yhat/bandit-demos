from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback, LambdaCallback
import numpy
import os

from bandit import Bandit

bandit = Bandit('colin', 'c4548110-cc4b-11e6-a5c5-0242ac110003','http://54.201.192.120/')

# bandit = Bandit()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset = numpy.loadtxt(os.path.join(dir_path, "pima-indians-diabetes.csv"), delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Setup our loss curves
# plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: bandit.report('loss', float((logs['loss']))))
class BanditLogs(Callback):
    def on_epoch_end(self, batch, logs={}):
        print(logs)
        # bandit.report('loss', float(logs['loss']))
        # print(type(logs['loss']))

# plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(type(float(logs['loss']))))

# setup our history
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

banditlogs = BanditLogs()
history = LossHistory()

# Fit the model
model.fit(X, Y, nb_epoch=200, batch_size=10,  verbose=0, callbacks=[history, banditlogs])

bandit.metadata.loss = round(history.losses[-1].tolist(),5)
