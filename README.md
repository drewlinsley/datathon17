## Welcome to the NeuroDatathon, sponsored by the Brown Institute of Brain Science (BIBS) and the Computation in Brain and Mind Initiative (CBM).

## Prerequisites and schedule
    # You will compete with others to analyze our one-of-a-kind dataset of neural and behavior recodings. You can do this by yourself or in teams.

    All analyses must be done on the Brown CIS stronghold. If you have signed up for the competition, you will receive a login information email from CIS. See STRONGHOLD.md for details on logging in and using the stronghold environment.

    # There are two competition tracks. There will be two winners per track.
        (1) OPTIMIZE: your goal is to build the best machine learning model for decoding behavior from neural activity. There are 46 total behavioral variables to decode. You will receive a point for achieving the best decoding on a variable (as measured by pearson correlation between its predicted and ground truth values). The two teams that accrue the most points will win. 


        (2) CREATE, where you will analyze the dataset and come up with a novel analysis. Because this is a brand new dataset of neural activity and behavior, participants in this track have the opportunity to come up with scientifically significant findings. The two teams judged by our panel of 3 neuroscience and datascience experts as having the most impactful and creative analyses will win.

    # Prizes:
        - 1st place winners in each track receive a grubhub gift card for $50 and 2 hours of consulting for your choice of Machine Learning, Deep Learning, or Neuroscientific ventures.
        - 2nd place winners in each track receive 1 hour of consulting for your choice of Machine Learning, Deep Learning, or Neuroscientific ventures.

## Want to compete in a team? Sign up here:
    # https://goo.gl/forms/lha7ENCEBRN7OvYa2

## Data description
    # We have split our dataset of neural and behavioral data into separate sets for (1) training your model and (2) validating your model. A third set is only accessible to the conference organizers. This is the testing set, and will be used to rank competition participants (on the OPTIMIZE track).
        - Training data
        - Validation data
        - Testing data

##  How to submit your work for judging?
    # Optimize:
        - See the "optimize_" scripts for examples. The goal is to pass the "test" neural data through your model, and save your behavior predictions for evaluation by the competition organizers. To do this you must import the "c_utils" module, select a team/username, then package your performance with the following command:
        ```
        c_utils.package_test_predictions(
            team_name=team_name,
            data=test_preds)
        ```

    # Create:
        - See the "create_" scripts for examples. You will create a brief report of your analysis (a line or two) and a figure of your findings. 
        . To do this you must import the "c_utils" module, select a team/username, then package your performance with the following command:
        ```
        c_utils.savefig(team_name)  # This saves your figure
        c_utils.package_test_predictions(  # This saves your report
            team_name=team_name,
            data=report,
            create=True)
        ```
    # IMPORTANT: After finishing your analysis, put your newly created files into the date-appropriate folder in the submissions folder.

##  Ideas for analyses?
    # Optimize
        - Optimize a deep neural network for decoding behavior from neural activity.
            + Starter script: "optimize_keras_decoding.py" or "optimize_sklearn_decoding".
        - Optimize a deep recurrent neural network for decoding behavior from neural activity.
            + Starter script: "optimize_keras_recurrent_decoding.py".
    # Create
        - Certain behavioral states are easier to predict than others? 
            + Use scikit learn to split behavior and neural data up into different "states".
            + Train classifiers to decode behavior in each state and look at the cross-state prediction performance. What makes some states easier than others to predict?
            + Motivating research: https://www.nature.com/articles/nature11129
            + Starter script: "create_sklearn_cluster.py"

        - Can deep learning predict behavior?
            + Use an LSTM (recurrent neural network) to learn predictive "latent" components in behavior which can predict future behaviors. For instance, if a twitch of the arm predicts a large-scale change in the kind of behavior being performed. 
            + Motivating research: http://datta.hms.harvard.edu/documents/mmc13.pdf
            + Starter script: "create_behavior_prediction.py"

##  Tips and resources for fitting models:
    # Activation functions.
        - These are applied to the hidden activities in a neural network. Keras has a large list of these that you can easily swap into your model: https://keras.io/activations/
    # Model layers.
        - In keras, you construct your model by stacking layers on top of each other. The most commonly used operation in the example scripts is "Dense", which is a "fully-connected layer". What this means is that the model will learn weights that map all dimensions (i.e. columns) of your input data to all dimensions of the Dense layer (you specify this number by hand). The output of this layer, or the activities, are then passed to whatever layer you stack on top.
        - Other layers to try are 1D convolution (https://keras.io/layers/convolutional/), locally connected 1d (https://keras.io/layers/local/), and recurrent layers (https://keras.io/layers/recurrent/), the latter of which which expects sequences of data instead of vectors or matrices (see "optimize_keras_recurrent_decoding.py" for an example).
        - In scikit learn you will deal with "shallow" models to decode behavior from neural activity. See the following link for a large list of models to try (example script is "optimize_sklearn_decoding.py"): http://scikit-learn.org/stable/supervised_learning.html
    # Loss functions.
        - A loss function describes the error of your model's predictions, and is used to optimize the model's weights to reduce those errors in the future. In both keras and scikit-learn you can relatively easily screen a wide variety of these. A warning: think hard about the kind of error that a loss function is evaluating before trying it in this competition. For instance, some loss functions (like binary cross entropy) are designed for classification problems ("Is this a dog or a cat?"). But the data provided to you in this competition sets up a regression problem ("What is the price of this house given its square footage?"). The data must be consistent with the loss function!
        - Keras loss functions: https://keras.io/losses/
        - Scikit-learn loss functions: It's less straightforward to swap losses in and out of models, but an example of this can be found here (note that the loss is set to "hinge") http://scikit-learn.org/stable/modules/sgd.html
    # Optimizers.
        - After specifying the model, you must choose an optimizer that will train it for the task at hand. The chosen optimizer will adjust model weights so that they produce better predictions in the future. These can be easily swapped in and out of keras.
        - Keras optimizers: https://keras.io/optimizers/
    # Regularizers.
        - Regularizers control model overfitting, which happens when your model becomes very good at predicting training data but very poor at predicting data outside of your training set. This can be done by either augmenting your data in certain ways or adding mechanisms and special operations to your model. In keras and scikit learn the latter can be done relatively easily. Although it is less straightforward, huge performance gains could be achieved by experimenting with data augmention routines.
        - Keras regularizers:
            + https://keras.io/regularizers/
            + The typical way of regularizing your model is by adding weight decay (e.g. l2 kernel regularization). You can also try regularizing your model's activities. Another common method for regularization is with "dropout", where activities are randomly set to 0 (and you set the probability of this dropout).
        - Scikit-learn regularizers:
            + This depends on the model you are using. For instance, in the case of an SVM, the c-parameter regularizes the model http://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html

## General competition resources:

    # Deep learning resource
        - http://cs231n.stanford.edu/
    # Keras resource
        - https://keras.io/getting-started/sequential-model-guide/
    # Scikit learn resource
        - http://scikit-learn.org/stable/tutorial/statistical_inference/index.html
    # Neuroscience resource
        - https://arxiv.org/pdf/1708.00909.pdf
