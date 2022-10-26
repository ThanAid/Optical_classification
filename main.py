import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import arange, meshgrid
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve

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

# Step 2 -- print digit 131
lib.show_sample(X_train, 131)

# Step 3 -- plot all digits (random sample for each label)
lib.plot_digits_samples(X_train, y_train)

# Step 4 -- Calculate mean value of digit 0
mean_of_digit = lib.digit_mean_at_pixel(X_train, y_train, 0, pixel=(10, 10))
print('The mean value of the characteristics of the pixel(10,10) for the digit 0 (based on train data) is: ' +
      str(mean_of_digit) + '.')

# Step 5 -- Calculate variance of digit 0
variance_of_digit = lib.digit_variance_at_pixel(X_train, y_train, 0, pixel=(10, 10))
print('The variance of the characteristics of the pixel(10,10) for the digit 0 (based on train data) is: ' +
      str(variance_of_digit) + '.')

# Step 6 -- Calculate mean and variance of all pixels of digit 0
mean_of_digit = lib.digit_mean(X_train, y_train, 0)
variance_of_digit = lib.digit_variance(X_train, y_train, 0)

# Step 7 -- plot digit 0 using mean values calculated at step 6
plt.imshow(mean_of_digit.reshape(16, 16), cmap='gray')
plt.title('Digit 0 using mean values')
plt.show()

# Step 8 -- plot digit 0 using variance calculated at step 6
plt.imshow(variance_of_digit.reshape(16, 16), cmap='gray')
plt.title('Digit 0 using variance')
plt.show()

# Step 9 --  Calculate mean and variance of all pixels of all digits
mean_of_all = {}  # creating a dict to store the mean values for each digit e.g. mean_of_all = {digit: 16x16array}
var_of_all = {}  # creating a dict to store the variance for each digit e.g. var_of_all = {digit: 16x16array}

for i in range(10):
    mean_of_all[i] = lib.digit_mean(X_train, y_train, i)
    var_of_all[i] = lib.digit_variance(X_train, y_train, i)

fig = plt.figure(figsize=(12, 6))

for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(mean_of_all[i].reshape(16, 16), cmap='gray')
    ax.set_title('Digit ' + str(i))
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_yticks([0, 4, 8, 12, 16])

fig.suptitle("Plotting digits using mean values", fontweight="bold", fontsize=20)
plt.show()

# Step 10 and 11 -- Classification of element 101/ Classification of all elements from test data
preds = lib.euclidean_distance_classifier(X_test, mean_of_all)
print('\nThe element 101 of the test data is classified as:', preds[101])
print("The real class (digit) of element 101 is:", y_test[101])

success_rate = sum(preds == y_test) / len(preds)  # the number of successful predictions divided by the total number of
# predictions
print("\nThe success rate of the classifier is:", success_rate)

# Step 12 -- Created Euclidean Classifier as scikit-learn estimator
model = EuclideanDistanceClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
score_model = model.score(X_test, y_test)
print('The score of the Euclidian Classifier (created as a scikit_learn estimator) is', score_model)

# Step 13 -- score using 5 fold cross-validation
cross_score = lib.evaluate_classifier(model, X_train, y_train)
print(f'\nScore estimated via cross-validation with 5 folds is: {cross_score * 100}%.')

# transforming the data to 2D using PCA (to use it for plotting decision region) and training the model again
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)
X_train_2d = pca.fit_transform(X_train)

# re-training the model again using the reduced features
model_2d = EuclideanDistanceClassifier()
model_2d.fit(X_train_2d, y_train)

# Plot using created func
lib.plot_decision_region(model_2d, X_train_2d, y_train)

# Plotting learning curve
lib.plot_learning_curve(X_train,y_train)