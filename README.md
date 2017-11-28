# Welcome to the NeuroDatathon, sponsored by the Brown Institute of Brain Science (BIBS) and the Computation in Brain and Mind Initiative (CBM).

## Prerequisites and schedule
You will compete with others to analyze our one-of-a-kind dataset of neural and behavior recodings. You can do this by yourself or in teams.

All analyses must be done on the Brown CIS stronghold. If you have signed up for the competition, you will receive a login information email from CIS. See STRONGHOLD.md for details on logging in and using the stronghold environment.

There are two competition tracks. There will be two winners per track.

1. OPTIMIZE: your goal is to build the best machine learning model for decoding behavior from neural activity. There are 46 total behavioral variables to decode. You will receive a point for achieving the best decoding on a variable (as measured by pearson correlation between its predicted and ground truth values). The two teams that accrue the most points will win. 


2. CREATE, where you will analyze the dataset and come up with a novel analysis. Because this is a brand new dataset of neural activity and behavior, participants in this track have the opportunity to come up with scientifically significant findings. The two teams judged by our panel of 3 neuroscience and datascience experts as having the most impactful and creative analyses will win.

## Prizes:
1. 1st place winners in each track receive an Amazon gift card for $100.

2. 2nd place winners in each track receive an Amazon gift card for $50. (If participating in a team, prizes will be split amongst teammates.)

## Want to compete in a team? Sign up [here](https://goo.gl/forms/lha7ENCEBRN7OvYa2).

## Data description
We have split our dataset of neural and behavioral data into separate sets for (1) training and validating your model and (2) a test set that is only accessible to the conference organizers for evaluating competition participants on the OPTIMIZE track.
    
1. The training data is stored in a Python Numpy Dictionary. This has the extension `.npz`. It is loaded with: `np.load('data.npz')`, which gives a Python dictionary with the following keys: 
```
['val_behavior',
 'train_neural',
 'neural_idx',
 'train_behavior',
 'behavior_idx',
 'val_neural',
 'test_neural']
 ```
 The template scripts (`optimize_*.py` or `create_*.py`) provide examples for how to manipulate the data in the dictionary. Each dataset has neural or behavioral recordings, or an "index" that labels the different dimensions of each recording. For instance, the neural datasets have 22 dimensions (matrices with N rows and 22 columns), which are indexed by the aforementioned "index" variables.

 The `train_*` dataset variables (neural and behavior) each have 4447 rows. There are 22 neural dimensions and 46 behavior dimensions. It is up to you to manipulate those to build models for either optimizing behavior decoding from the neural activity or creating a novel analysis. For validation, it is preferable to cross-validate within this dataset (see the template scripts for examples).

2. After training a model, you will test the model with the `test_neural` dataset variable. The model's predictions for behavior corresponding to this neural activity will be used by the conference organizers for evaluation. See the template scripts for examples on (1) passing this data through your model and (2) producing a file with your predictions.

## How to submit your work for judging?
1. Optimize:
See the "optimize_" scripts for examples. As mentioned above, you will pass "test" neural data through your mode  and save your behavior predictions for evaluation by the competition organizers. To do this you must import the "c_utils" module, select a team/username, then package your performance with the following command:
```
c_utils.package_test_predictions(
    team_name=team_name,
    data=test_preds)
```
If successful, your script will save a file named something like `OPTIMIZE_11_27_purkinje.npy` (constructed as: `{Competition track}_{Month}_{Day}_{Team name}.npy`). This file will be automatically moved to the competition submissions folder for the day. You can overwrite your submission as many times as you'd like. The final submission will be evaluated the next morning. 

2. Create:
See the `create_` scripts for examples. You will create a brief report of your analysis (a line or two) and a figure of your findings. As with the optimize scripts these will be transfered to the submissions folder upon completion.

IMPORTANT: Your analysis files must be placed into the date-appropriate folder in the submissions folder. This should happen automatically, but make sure this is the case.

--------

### Ideas for analyses?
1. Optimize -- behavior decoding from neural activity.
Optimize a deep neural network for decoding behavior from neural activity.
    Starter script: `optimize_keras_decoding.py` or `optimize_sklearn_decoding`.
Optimize a deep recurrent neural network for decoding behavior from neural activity.
    Starter script: `optimize_keras_recurrent_decoding.py`.

2. Create -- Identify distinct neural representational states of different behaviors.
Certain behavioral states are easier to predict than others? 
Use scikit learn to split behavior and neural data up into different "states".
Train classifiers to decode behavior in each state and look at the cross-state prediction performance. What makes some states easier than others to predict?
[Motivating research](https://www.nature.com/articles/nature11129)
    Starter script: `create_sklearn_cluster.py`

3. Can deep learning predict behavior?
Use an LSTM (recurrent neural network) to learn predictive "latent" components in behavior which can predict future behaviors. For instance, if a twitch of the arm predicts a large-scale change in the kind of behavior being performed. 
[Motivating research](http://datta.hms.harvard.edu/documents/mmc13.pdf)
    Starter script: `create_behavior_prediction.py`

### Tips and resources for fitting models:
1. Activation functions.
These are applied to the hidden activities in a neural network. Keras has a large list of these that you can easily swap into your model: [https://keras.io/activations/](https://keras.io/activations/)

2. Model layers.
In keras, you construct your model by stacking layers on top of each other. The most commonly used operation in the example scripts is "Dense", which is a "fully-connected layer". What this means is that the model will learn weights that map all dimensions (i.e. columns) of your input data to all dimensions of the Dense layer (you specify this number by hand). The output of this layer, or the activities, are then passed to whatever layer you stack on top.
Other layers to try are [1D convolution](https://keras.io/layers/convolutional/), [locally connected 1d](https://keras.io/layers/local/), and [recurrent layers](https://keras.io/layers/recurrent/), the latter of which which expects sequences of data instead of vectors or matrices (see "optimize_keras_recurrent_decoding.py" for an example).
In scikit learn you will deal with "shallow" models to decode behavior from neural activity. See the following link for a large list of models to try (example script is `optimize_sklearn_decoding.py`): [http://scikit-learn.org/stable/supervised_learning.html](http://scikit-learn.org/stable/supervised_learning.html)

3. Loss functions.
A loss function describes the error of your model's predictions, and is used to optimize the model's weights to reduce those errors in the future. In both keras and scikit-learn you can relatively easily screen a wide variety of these. A warning: think hard about the kind of error that a loss function is evaluating before trying it in this competition. For instance, some loss functions (like binary cross entropy) are designed for classification problems ("Is this a dog or a cat?"). But the data provided to you in this competition sets up a regression problem ("What is the price of this house given its square footage?"). The data must be consistent with the loss function!
[Keras loss functions](https://keras.io/losses/)
[Scikit-learn loss functions](http://scikit-learn.org/stable/modules/sgd.html): It's less straightforward to swap losses in and out of models, but an example of this can be found here (note that the loss is set to "hinge")

4. Optimizers.
After specifying the model, you must choose an optimizer that will train it for the task at hand. The chosen optimizer will adjust model weights so that they produce better predictions in the future. These can be easily swapped in and out of keras.
[Keras optimizers](https://keras.io/optimizers/)

5. Regularizers.
Regularizers control model overfitting, which happens when your model becomes very good at predicting training data but very poor at predicting data outside of your training set. This can be done by either augmenting your data in certain ways or adding mechanisms and special operations to your model. In keras and scikit learn the latter can be done relatively easily. Although it is less straightforward, huge performance gains could be achieved by experimenting with data augmention routines.
[Keras regularizers](https://keras.io/regularizers/): The typical way of regularizing your model is by adding weight decay (e.g. l2 kernel regularization). You can also try regularizing your model's activities. Another common method for regularization is with "dropout", where activities are randomly set to 0 (and you set the probability of this dropout).
[Scikit-learn regularizers](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html): This depends on the model you are using. For instance, in the case of an SVM, the c-parameter regularizes the model 

### General competition resources:
1. [Deep learning resource](http://cs231n.stanford.edu/)
2. [Keras resource](https://keras.io/getting-started/sequential-model-guide/)
3. [Scikit learn resource](http://scikit-learn.org/stable/tutorial/statistical_inference/index.html)
4. [Neuroscience decoding resource](https://arxiv.org/pdf/1708.00909.pdf)

