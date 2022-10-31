import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import lib  # import lib.py that contains functions created for this exercise
from lib import EuclideanDistanceClassifier

# Step 1 -- Read data and assign values.
test_data = pd.read_csv('test.txt', sep=' ', header=None)
train_data = pd.read_csv('train.txt', sep=' ', header=None)
test_data = test_data.drop([257], axis=1)  # last column was containing NaN values
train_data = train_data.drop([257], axis=1)  # last column was containing NaN values

y_test = test_data[0].to_numpy()
X_test = test_data.drop([0], axis=1).to_numpy()
y_train = train_data[0].to_numpy()
X_train = train_data.drop([0], axis=1).to_numpy()

print('Data uploaded.')

# Step 14 -- calculating a-priors
counted = lib.calculate_priors(y_train)
print('a priors:', counted)
# TODO bar graph

# Step 15 -- Naive Bayesian Classifier
model = lib.CustomNBClassifier()
model.fit(X_train, y_train)
