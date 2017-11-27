"""Per-joint decoding with keras."""

import os
import numpy as np
import c_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, regularizers


# FILES
data_file = os.path.join(
    'data',
    'raw_datathon_data.npz')

# VARS
normalize_data = True
team_name = 'purkinje'  # Enter your username or team name.
# Your results file will be tagged with this.
l2_wd = 0.001  # Use l2 regularization on deep network weights
lr = 1e-3  # Learning rate
loss_fun = 'logcosh'  # Use this kind of loss
epochs = 50  # Train for this many epochs
l1_weights = 64  # Use this many weights in the first layer
l2_weights = 128  # Use this many weights in the second layer
l1_dropout = 0.1  # Dropout applied to layer 1 activities
l2_dropout = 0.1  # Dropout applied to layer 2 activities
activation = 'relu'  # Use this kind of nonlinear activation function

# Load neural data
raw_data = np.load(data_file)  # Load a saved numpy dictionary
train_proc_behavior = raw_data['train_behavior']  # Split off train behavior
train_proc_neural = raw_data['train_neural']  # Split off train neural
test_proc_neural = raw_data['test_neural']  # Split off test neural
behavior_labels = raw_data['behavior_idx']
neural_labels = raw_data['neural_idx']

if normalize_data:  # normalize features to 0 mean and unit variance
    # Neural data
    mu_n = train_proc_neural.mean(0).reshape(1, -1)
    sd_n = train_proc_neural.std(0).reshape(1, -1) + 1e-12
    train_proc_neural = (train_proc_neural - mu_n) / sd_n
    test_proc_neural = (test_proc_neural - mu_n) / sd_n
    # Behavior data
    mu_b = train_proc_behavior.mean(0).reshape(1, -1)
    sd_b = train_proc_behavior.std(0).reshape(1, -1) + 1e-12
    train_proc_behavior = (train_proc_behavior - mu_b) / sd_b

# An idea -- remove "uniformative" neurons/columns from
# train_proc_neural and val_proc_neural.
# This is called "feature selection" and can improve
# decoding performance.

# Construct and compile model
model = Sequential()
reg = regularizers.l2(l2_wd)
model.add(
    Dense(
        l1_weights,
        activation=activation,
        kernel_regularizer=reg,
        input_dim=train_proc_neural.shape[-1]))
model.add(Dropout(l1_dropout))
model.add(
    Dense(
        l2_weights,
        activation=activation,
        kernel_regularizer=reg))
model.add(Dropout(l2_dropout))
model.add(
    Dense(
        train_proc_behavior.shape[-1],
        activation='linear'))
adam = optimizers.adam(lr=lr)
model.compile(
    optimizer=adam,
    loss=loss_fun)

# Train model
val_proc_neural = train_proc_neural[-100:, :]
val_proc_behavior = train_proc_behavior[-100:, :]
train_proc_neural = train_proc_neural[:-100, :]
train_proc_behavior = train_proc_behavior[:-100, :]
losses = model.fit(
    train_proc_neural,
    train_proc_behavior,
    epochs=epochs,
    validation_split=0.1)

# Make predictions and measure correlation on validation set
preds = model.predict(
    val_proc_neural)  # Make predictions on validation data
perf = c_utils.model_performance(  # Calculate performance
    preds=preds,
    gt=val_proc_behavior,
    labels=behavior_labels)

# Make predictions on the test set and save for the competition
test_preds = model.predict(test_proc_neural)
fn = c_utils.package_test_predictions(
    team_name=team_name,
    data=test_preds)
c_utils.movefile_for_eval(fn)

