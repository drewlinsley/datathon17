"""Per-joint decoding with keras."""

import os
import numpy as np
import c_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers
from matplotlib import pyplot as plt


# FILES
data_file = os.path.join(
    'data',
    'raw_datathon_data.npz')

# MODEL VARS
team_name = 'purkinje'  # Enter your username or team name.
# Your results file will be tagged with this.
delay = 3  # Break behavior into chunks this long
LSTM_weights = 128  # Use this many Recurrent weights to predict behavior
normalize_chunks = 1  # Split data into this many chunks for normalization
f_dropout = 0.2  # Dropout on the feed-forward LSTM activity
r_dropout = 0.2  # Dropout on the recurrent LSTM activity
lr = 1e-3  # Learning rate
decay = 1e-6  # Learning rate decay
momentum = 0.9  # Learning rate momentum
nesterov = True  # Use Nesterov's mometum correction
loss_fun = 'logcosh'  # Use a Logarithm of the hyperbolic cosine loss
epochs = 10  # Train for this many epochs
activation = 'relu'  # Use this kind of nonlinear activation function

# Load behavior data
raw_data = np.load(data_file)  # Load a saved numpy dictionary
train_proc_behavior = raw_data['train_behavior']  # Split off train behavior
val_proc_behavior = raw_data['val_behavior']  # Split off val behavior
behavior_labels = raw_data['behavior_idx']

# Get delay-staggered matrix of proc_behavior
train_future_behavior = c_utils.delay_mat(
    train_proc_behavior, delay)  # Process train behavior -> LSTM-ready tensor
val_future_behavior = c_utils.delay_mat(
    val_proc_behavior, delay)  # Process val behavior -> LSTM-ready tensor

# Construct and compile model
model = Sequential()
model.add(
    LSTM(  # Add an LSTM recurrent layer
        LSTM_weights,
        dropout=f_dropout,
        recurrent_dropout=r_dropout,
        input_shape=(delay, train_future_behavior.shape[-1])))
model.add(
    Dense(  # Add a fully connected prediction layer
        train_future_behavior.shape[-1], activation=activation))
sgd = optimizers.SGD(  # Add a gradient descent optimizer
    lr=lr,
    decay=decay,
    momentum=momentum,
    nesterov=nesterov)
model.compile(  # Compile your model
    optimizer=sgd,
    loss=loss_fun)
losses = model.fit(  # Train the model
    train_future_behavior,
    train_proc_behavior,
    epochs=epochs,
    validation_data=(val_future_behavior, val_proc_behavior))
preds = model.predict(
    val_future_behavior)  # Make predictions on validation data
perf = c_utils.model_performance(  # Calculate performance
    preds=preds,
    gt=val_proc_behavior,
    labels=behavior_labels)

# Extract LSTM weight matrices, which are organized I/F/C/O
f_weights = c_utils.split_kernels(
    x=model.layers[0].get_weights()[0],
    splits=4,
    num_weights=LSTM_weights)
r_weights = c_utils.split_kernels(
    x=model.layers[0].get_weights()[1],
    splits=4,
    num_weights=LSTM_weights)
finput, fforget, fcell, foutput = f_weights
rinput, rforget, rcell, routput = r_weights

# Look at feedforward weight similarities. ID similar behavior dimensions!
fidist = c_utils.sq_dist(finput)
ffdist = c_utils.sq_dist(fforget)
fcdist = c_utils.sq_dist(fcell)
fodist = c_utils.sq_dist(foutput)

# Repeat with the recurrent matrices
ridist = c_utils.sq_dist(rinput)
rfdist = c_utils.sq_dist(rforget)
rcdist = c_utils.sq_dist(rcell)
rodist = c_utils.sq_dist(routput)

# Visualize everything
f_dists = [
    fidist,
    ffdist,
    fcdist,
    fodist
]
r_dists = [
    fidist,
    ffdist,
    fcdist,
    fodist
]
labels = ['Input', 'Forget', 'Cell', 'output']
c_utils.plot_weights(
    xs=f_dists,
    ls=labels,
    save_plot=None)
c_utils.plot_weights(
    xs=r_dists,
    ls=labels,
    save_plot=None)
plt.suptitle('Recurrent matix distances.')

# Write a brief report and save associated data.
report = 'We trained a recurrent model to predict behavior. ' +\
    'With this model we found that certain latend states predict ' +\
    'certain behaviors.'  # Obviously, expand on this analysis.

# Make predictions on the test set and save for the competition
fn = c_utils.savefig(team_name)
c_utils.movefile_for_eval(fn)
fn = c_utils.package_test_predictions(
    team_name=team_name,
    data=report,
    create=True)
c_utils.movefile_for_eval(fn)
plt.show()
