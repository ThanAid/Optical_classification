import itertools
import math
import random
import statistics
import numpy as np
import pandas as pd
from numpy import arange, meshgrid
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from scipy.stats import multivariate_normal


def show_sample(X, index):
    """Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    """

    arr = np.reshape(X[index], (16, 16))
    plt.imshow(arr, cmap='gray')
    plt.title(f'Digit with index number: {index}', fontsize=20, fontweight="bold")
    plt.show()


def plot_digits_samples(X, y):
    """Takes a dataset and selects one example from each label and plots it in subplots
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    """
    fig = plt.figure(figsize=(12, 6))
    y_df = pd.DataFrame(y)  # Convert to df to find the index of each digit in the loop below

    for i in range(10):
        random_num = random.choice(y_df[y_df[0] == i].index.values)  # finds a random index for each digit
        digit = np.reshape(X[random_num], (16, 16))  # locates the row picked by the
        # random choice for that specific digit and converts it to numpy array
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(digit, cmap='gray')
        ax.set_title('Digit ' + str(i))
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_yticks([0, 4, 8, 12, 16])

    plt.suptitle("Plotting a random sample for each digit", fontsize=20, fontweight="bold")
    plt.show()


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the mean for all instances of a specific digit at a pixel location
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.
    Returns:
        (float): The mean value of the digits for the specified pixels
        :param X:
        :param y:
        :param digit:
        :param pixel:
    """
    sum_pixel = 0
    x_digit = X[y == digit]  # Find the samples of features for the specified digit

    for i in range(len(x_digit)):
        arr = np.reshape(x_digit[i], (16, 16))
        sum_pixel += arr[pixel[0], pixel[1]]

    return sum_pixel / (len(x_digit))


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the variance for all instances of a specific digit at a pixel location
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select
    Returns:
        (float): The variance value of the digits for the specified pixels
        :param pixel:
    """
    pixel_list = []
    x_digit = X[y == digit]  # Find the samples of features for the specified digit

    for i in range(len(x_digit)):
        arr = np.reshape(x_digit[i], (16, 16))
        pixel_list.append(arr[pixel[0], pixel[1]])

    return statistics.variance(pixel_list)


def digit_mean(X, y, digit):
    """Calculates the mean for all instances of a specific digit
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    """
    X_digit = X[y == digit]
    mean_array = np.mean(X_digit, axis=0)

    return mean_array


def digit_variance(X, y, digit):
    """Calculates the variance for all instances of a specific digit
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    """
    X_digit = X[y == digit]
    var_array = np.var(X_digit, axis=0)

    return var_array


def euclidean_distance(s, m):
    """Calculates the euclidean distance between a sample s and a mean template m
    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)
    Returns:
        (float) The Euclidean distance between s and m
    """

    dist_array = abs(s - m)
    ssd = 0  # sum of squares of Euclidian distance for each pixel of the digit
    for i in np.nditer(dist_array):
        ssd += i ** 2

    return math.sqrt(ssd)


def euclidean_distance_classifier(X, X_mean):
    """Classifiece based on the euclidean distance between samples in X and template vectors in X_mean
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)
    Returns:
        (np.ndarray) predictions (nsamples)
    """
    eucl_dist = {}  # creating a dict to store the euclidean distances between samples in X and X_mean e.g.
    # eucl_dist = {digit: eucl dist (float)}
    preds = []

    for i in range(len(X)):  # Calculating how many digits we have stored in X
        for j in range(10):
            eucl_dist[j] = euclidean_distance(X[i], X_mean[j])  # evaluate euclidian distance for each digit
        preds.append(min(eucl_dist, key=eucl_dist.get))  # append the distance for that specific test digit

    return np.array(preds)


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        mean_arr = np.empty([10, X.shape[1]])
        for i in range(10):
            mean_arr[i] = digit_mean(X, y, i)

        self.X_mean_ = mean_arr
        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """

        return euclidean_distance_classifier(X, self.X_mean_)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        return sum(self.predict(X) == y) / len(y)


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y
    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        folds (int): the number of folds
    Returns:
        (float): The 5-fold classification mean score and std(accuracy)
    """
    results = cross_val_score(estimator=clf, X=X, y=y, cv=folds)
    accur = results.mean()
    accur_std = results.std()
    return accur, accur_std


def calculate_priors(y):
    """Return the a-priori probabilities for every class
    Args:
        y (np.ndarray): Labels for dataset (nsamples)
    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    counted = []
    n = len(y)

    for digit in range(10):
        counted.append(np.count_nonzero(y == digit))

    return np.array([a / n for a in counted])


class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier
     var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability."""

    def __init__(self, use_unit_variance=False, var_smoothing=1e-9):
        self.use_unit_variance = use_unit_variance
        self.X_mean_ = None
        self.X_var_ = None
        self.priors_ = None
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        Calculates self.X_var_ based on the variance
        feature values in X for each class.
        self.X_mean_  and self.X_var_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        mean_arr = np.empty([10, X.shape[1]])
        var_arr = np.empty_like(mean_arr)

        for i in range(10):
            mean_arr[i] = digit_mean(X, y, i)  # Calculate the mean values of features
            var_arr[i] = digit_variance(X, y, i)  # Calculate the variance values of features

        counted = calculate_priors(y)

        self.X_mean_ = mean_arr
        if not self.use_unit_variance:
            self.X_var_ = var_arr
        else:
            self.X_var_ = np.ones_like(
                var_arr)  # if use_unit_variance is True then the variance values will be set as 1
        self.priors_ = counted

        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        Bayes Theorem
        Args:
            X (np.ndarray): Digits data (nsamples x nfeatures)
        """
        preds = []
        # adding a small value to the variance values of the features to avoid being treated as 0 if they are too small
        var = self.X_var_ + (self.var_smoothing * self.X_var_.max())

        for i in range(len(X)):  # Calculating how many digits we have stored in X
            posterior = []  # list that will store posterior

            for digit in range(10):
                pxc = np.prod((1 / (np.sqrt((2 * np.pi * var[digit])))) * np.exp(  # Calculating the Π(P(x|y))
                    -np.power(X[i] - self.X_mean_[digit], 2) / (2 * var[digit])))

                posterior.append(np.log(pxc) + np.log(self.priors_[digit]))  # appending the result from discriminative
                # function for that digit (possibility for that sample to be classified as that digit)

            preds.append(posterior.index(max(posterior)))  # append the index of the max value of the posterior list,
            # meaning that the sample gets classified as the digit with the max likelihood

        return np.array(preds)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        return sum(self.predict(X) == y) / len(y)


class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: initialize model, criterion and optimizer
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        raise NotImplementedError

    def fit(self, X, y):
        # TODO: split X, y in train and validation set and wrap in pytorch dataloaders
        train_loader = ...
        val_loader = ...
        # TODO: Train model
        raise NotImplementedError

    def predict(self, X):
        # TODO: wrap X in a test loader and evaluate
        test_loader = ...
        raise NotImplementedError

    def score(self, X, y):
        # Return accuracy score.
        raise NotImplementedError


def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_custom_nb_classifier(clf, X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier"""

    raise NotImplementedError


def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def plot_decision_region(model, X, y):
    """Plots the Decision region of a model
        Args:
            model (sklearn.base.BaseEstimator): classifier
            X (np.ndarray): Digits data (nsamples x nfeatures), where nfeatures = 2
            y (np.ndarray): Labels for dataset (nsamples)
    """
    fig, ax = plt.subplots()

    # define bounds of the domain
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    # define the x and y scale
    x1grid = arange(min1, max1, 0.1)
    x2grid = arange(min2, max2, 0.1)

    # create all of the lines and rows of the grid
    xx, yy = meshgrid(x1grid, x2grid)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, alpha=0.8)

    # Creating Decision Region
    for i in range(10):
        dots = ax.scatter(X[:, 0][y == i], X[:, 1][y == i], label='Digit ' + str(i), s=60,
                          alpha=0.9, edgecolors='k')

    ax.set_ylabel('Feature 1')
    ax.set_xlabel('Feature 2')
    ax.set_title('Decision Surface')
    ax.legend()
    plt.show()


def plot_learning_curve(X, y):
    """Plots the Learning curve of a model
            Args:
                X (np.ndarray): Digits data (nsamples x nfeatures)
                y (np.ndarray): Labels for dataset (nsamples)
        """
    # we are using Learning curve to get train_sizes, train_score and test_score
    train_sizes, train_scores, test_scores = learning_curve(EuclideanDistanceClassifier(), X, y, cv=5,
                                                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    # Now we calculate the mean and standard deviation of the train and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plot the learning curve
    plt.subplots(1, figsize=(10, 10))
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Plots the confusion matrix of the evaluation of the class for a specific digit
                Args:
                    cmap:
                    title:
                    cm (np.ndarray): confusion matrix from the test data and the predictions (n_predictedclasses x n_actualclasses)
                    classes (list): unique classes of the dataset (n_actualclasses)
            """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def bar_plot(x, y, title='title', x_label='x', y_label='y', LABELS=None):
    """Plots a bar plot
                Args:
                    x (list): values used on x axes
                    y (list): values used on y axes
                    title (str)
                    x_label (str)
                    y_label (str)
                    LABELS (list): custom labels for x axis
            """
    # creating the bar plot
    fig = plt.bar(x, y, color='maroon',
                  width=0.5)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, fontsize=16)
    plt.bar_label(fig, labels=['%.3f' % e for e in y])
    plt.xticks(x, LABELS)
    if LABELS is not None: plt.xticks(rotation=45, ha='right', fontsize=6, fontweight='bold')
    plt.show()


def var_smooth_score(X, y, X_valid, y_valid, start_value=1e-9, step=2, log=False):
    """Plots a line plot about var smoothing values for the Custom NBC model
            Args:
                X (np.ndarray): Digits data (nsamples x nfeatures)
                y (np.ndarray): Labels for dataset (nsamples)
                X_valid (np.ndarray): Digits data (nsamples x nfeatures)
                y_valid (np.ndarray): Labels for dataset (nsamples)
                start_value (int): the starting value for the smoother
                step (int): step of multipl for start_value
                log (bool): if True then x axis is converted to log values
                """
    temp_smoothie = []
    smoothies = {}
    while start_value < 0.5:
        temp_smoothie.append(start_value)
        start_value *= step
    for var_smoother in temp_smoothie:
        model = CustomNBClassifier(var_smoothing=var_smoother)
        model.fit(X, y)
        preds = model.predict(X_valid)
        smoothies[var_smoother] = sum(preds == y_valid)

    y_ax = smoothies.values()
    if not log:
        x_ax = smoothies.keys()
    else:
        x_ax = [math.log(k) for k in smoothies.keys()]

    plt.plot(x_ax, y_ax)
    plt.title('Var smoother Score', fontsize=16)
    plt.xlabel('Smoother value')
    plt.show()
