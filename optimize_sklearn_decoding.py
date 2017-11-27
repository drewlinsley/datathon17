"""Per-joint decoding with keras."""

import os
import c_utils
import numpy as np
from sklearn import svm


# FILES
data_file = os.path.join(
    'data',
    'raw_datathon_data.npz')

# VARS
normalize_data = True
team_name = 'purkinje'  # Enter your username or team name.

# Load neural data
raw_data = np.load(data_file)  # Load a saved numpy dictionary
train_proc_behavior = raw_data['train_behavior']  # Split off train behavior
val_proc_behavior = raw_data['val_behavior']  # Split off val behavior
train_proc_neural = raw_data['train_neural']  # Split off train neural
val_proc_neural = raw_data['val_neural']  # Split off val neural
test_proc_neural = raw_data['test_neural']  # Split off test neural
behavior_labels = raw_data['behavior_idx']
neural_labels = raw_data['neural_idx']

if normalize_data:  # normalize features to 0 mean and unit variance
    # Neural data
    mu_n = train_proc_neural.mean(0).reshape(1, -1)
    sd_n = train_proc_neural.std(0).reshape(1, -1) + 1e-12
    train_proc_neural = (train_proc_neural - mu_n) / sd_n
    val_proc_neural = (val_proc_neural - mu_n) / sd_n
    test_proc_neural = (test_proc_neural - mu_n) / sd_n
    # Behavior data
    mu_b = train_proc_behavior.mean(0).reshape(1, -1)
    sd_b = train_proc_behavior.std(0).reshape(1, -1) + 1e-12
    train_proc_behavior = (train_proc_behavior - mu_b) / sd_b
    val_proc_behavior = (val_proc_behavior - mu_b) / sd_b

# An idea -- remove "uniformative" neurons/columns from
# train_proc_neural and val_proc_neural.
# This is called "feature selection" and can improve
# decoding performance.

# Construct and compile model in loops
perfs = []
for selected_behavior in range(train_proc_behavior.shape[-1]):
    clf = svm.SVR()
    clf.fit(train_proc_neural, train_proc_behavior[:, selected_behavior])

    # Make predictions and measure correlation
    preds = clf.predict(
        val_proc_neural)  # Make predictions on validation data
    perf = np.corrcoef(val_proc_behavior[:, selected_behavior], preds)[0, 1]
    if selected_behavior % 2 == 0:
        print '%s X Validation correlation is %s' % (
            selected_behavior,
            perf)
    else:
        print '%s Y Validation correlation is %s' % (
            selected_behavior,
            perf)
    perfs += [perf]

# Make predictions on the test set and save for the competition
test_preds = clf.predict(test_proc_neural)
fn = c_utils.package_test_predictions(
    team_name=team_name,
    data=test_preds)
c_utils.movefile_for_eval(fn)

