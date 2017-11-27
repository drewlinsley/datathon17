"""Per-joint decoding with keras."""

import os
import c_utils
import numpy as np
from sklearn import manifold, cluster
from matplotlib import pyplot as plt
from matplotlib import cm


# FILES
data_file = os.path.join(
    'data',
    'raw_datathon_data.npz')

# VARS
team_name = 'purkinje'  # Enter your username or team name.
# Your results file will be tagged with this.

# Load neural data
raw_data = np.load(data_file)  # Load a saved numpy dictionary
train_proc_behavior = raw_data['train_behavior']  # Split off train behavior
val_proc_behavior = raw_data['val_behavior']  # Split off val behavior
train_proc_neural = raw_data['train_neural']  # Split off train neural
val_proc_neural = raw_data['val_neural']  # Split off val neural
behavior_labels = raw_data['behavior_idx']
neural_labels = raw_data['neural_idx']

mu_n = train_proc_neural.mean(0).reshape(1, -1)
sd_n = train_proc_neural.std(0).reshape(1, -1) + 1e-12
train_proc_neural = (train_proc_neural - mu_n) / sd_n
val_proc_neural = (val_proc_neural - mu_n) / sd_n

mu_b = train_proc_behavior.mean(0).reshape(1, -1)
sd_b = train_proc_behavior.std(0).reshape(1, -1) + 1e-12
train_proc_behavior = (train_proc_behavior - mu_b) / sd_b
val_proc_behavior = (val_proc_behavior - mu_b) / sd_b

# Use a cluster analysis to split train behavior and neural
# data into distinct "groups".
af_clf = cluster.AffinityPropagation()
neural_clusters = af_clf.fit_predict(train_proc_neural)
behavior_clusters = af_clf.fit_predict(train_proc_behavior)

# Loop through all clusters and calculate distances
neural_combos = []
for nc in np.unique(neural_clusters):
    it_data = train_proc_neural[neural_clusters == nc, :].mean(0)
    neural_combos += [it_data]
neural_combo_data = np.asarray(neural_combos)
neural_state_matrix = c_utils.sq_dist(neural_combo_data)
behavior_combos = []
for bc in np.unique(behavior_clusters):
    it_data = train_proc_neural[behavior_clusters == bc, :].mean(0)
    behavior_combos += [it_data]
behavior_combo_data = np.asarray(behavior_combos)
behavior_state_matrix = c_utils.sq_dist(behavior_combo_data)

# Plot distances in the state matrices -- some are more similar to others
f, axs = plt.subplots(2)
axs[0].imshow(neural_state_matrix)
axs[1].imshow(behavior_state_matrix)
plt.show()
plt.close(f)

# Construct and compile behavior dimension model
bc_clf = manifold.TSNE(n_components=2)
bc_embedded = bc_clf.fit_transform(train_proc_behavior.transpose())

# Construct and compile behavior event model
be_clf = manifold.TSNE(n_components=2)
be_embedded = be_clf.fit_transform(train_proc_behavior)

# Construct and compile neural dimension model
nc_clf = manifold.TSNE(n_components=2)
nc_embedded = nc_clf.fit_transform(train_proc_neural.transpose())

# Construct and compile neural event model
ne_clf = manifold.TSNE(n_components=2)
ne_embedded = nc_clf.fit_transform(train_proc_neural)

# Plot both embeddings in the same figure
f, axs = plt.subplots(3)
for idx, bc in enumerate(
        np.unique(behavior_clusters)):
    axs[0].scatter(
        be_embedded[behavior_clusters == bc, 0],
        be_embedded[behavior_clusters == bc, 1],
        c=cm.Reds_r(idx))
axs[0].set_title('Behavior embeddings colored by state.')

for idx, nc in enumerate(
        np.unique(neural_clusters)):
    axs[1].scatter(
        ne_embedded[neural_clusters == nc, 0],
        ne_embedded[neural_clusters == nc, 1],
        c=cm.Greens_r(idx))
axs[1].set_title('Neural embeddings colored by state.')
axs[2].plot(be_embedded[:, 0], be_embedded[:, 1], 'r', alpha=1)
axs[2].plot(ne_embedded[:, 0], ne_embedded[:, 1], 'g', alpha=0.6)

# Write a brief report and save associated data.
report = 'We applied a cluster analysis to behavior and neural ' +\
    'data, identifying "states" in each. There are different ' +\
    'numbers of clusters in each. This suggests that better ' +\
    'decoding performance could be achieved by selecting ' +\
    'the behavior and neural dimensions that best harmonize ' +\
    'the two data sources.'

# Make predictions on the test set and save for the competition
fn = c_utils.savefig(team_name)
c_utils.movefile_for_eval(fn)
fn = c_utils.package_test_predictions(
    team_name=team_name,
    data=report,
    create=True)
c_utils.movefile_for_eval(fn)
plt.show()
plt.close(f)
