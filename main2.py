import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import lib  # import lib.py that contains functions created for this exercise
from lib import EuclideanDistanceClassifier
import warnings  # TODO na to vgaloume?

warnings.filterwarnings("ignore")

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
print('\n------------------a priors------------------\n', counted)
lib.bar_plot([i for i in range(10)], counted, title='a-prior for each Digit', x_label='Digit', y_label='%')  # bar plot
print('--------------------------------------------')


# Step 15 -- Naive Bayesian Classifier
print('\n--------------Custom Model--------------')
model = lib.CustomNBClassifier()
model.fit(X_train, y_train)  # fitting the model
score = model.score(X_test, y_test)  # model score
print('The score of the custom model is:', score * 100, '%.')

cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train)  # 5-fold cv score
print(f'\nScore estimated via cross-validation for Custom NB with 5 folds is:'
      f' {cross_score * 100} \u00B1 {score_std * 100}%.')
print('--------------------------------------------')

print('\n--------------Gaussian Model--------------')
# Now GaussianNB will be used and compared to the custom above
model = GaussianNB()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # model score
print('The score of the Gaussian NB model is:', score * 100, '%.')

cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train)  # 5-fold cv score
print(f'\nScore estimated via cross-validation for Gaussian NB with 5 folds is:'  
      f' {cross_score * 100} \u00B1 {score_std * 100}%.')
print('--------------------------------------------')


# Step 16 -- Setting variance values "1" and re-training the model
print('\n--------------Custom Model--------------')
print('Setting variance 1 for all features')
model = lib.CustomNBClassifier(use_unit_variance=True)
model.fit(X_train, y_train)  # fitting the model
score = model.score(X_test, y_test)  # model score
print('The score of the custom model is:', score * 100, '%.')

cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train)  # 5-fold cv score
print(f'\nScore estimated via cross-validation for Custom NB with 5 folds is:'
      f' {cross_score * 100} \u00B1 {score_std * 100}%.')
print('-------------------------------------------------------')


# Step 17 -- Naive Bayes, Nearest Neighbors, SVM comparison
# Creating all models needed and appending them to a list of models
models = []
model_custom_nb = lib.CustomNBClassifier()
models.append(model_custom_nb)
model_nearest_n1 = KNeighborsClassifier(n_neighbors=1)
models.append(model_nearest_n1)
model_nearest_n3 = KNeighborsClassifier(n_neighbors=3)
models.append(model_nearest_n3)
model_svm_linear = SVC(kernel='linear')
models.append(model_svm_linear)
model_svm_poly = SVC(kernel='poly')
models.append(model_svm_poly)
model_svm_rbf = SVC(kernel='rbf')
models.append(model_svm_rbf)
model_svm_sigmoid = SVC(kernel='sigmoid')
models.append(model_svm_sigmoid)

# fit models
for mod in models:
    mod.fit(X_train, y_train)


# evaluating models' score
print('\n-------------------Model scores-----------------------')
model_dict = {}  # dictionary to store scores for each model
for mod in models:
    model_dict[mod] = mod.score(X_test, y_test) * 100
    print(f'Model "{mod}" has score {model_dict[mod]}%')

lib.bar_plot([i for i in range(7)], model_dict.values(), title="Model score", x_label='Model', y_label='%', LABELS=
['CustomNBC', 'Kn1', 'Kn3', 'SVC(lin)', 'SVC(poly)', 'SVC', 'SVC(sigm)'])  # bar plot the results

print('--------------------------------------------------------')

# evaluating models' 5 fold CV
print('\n-------------------Model 5-fold CV-----------------------')
model_dict5 = {}
for mod in models:
    model_dict5[mod] = lib.evaluate_classifier(mod, X_train, y_train)
    print(f'Model "{mod}" has 5 fold CV {model_dict5[mod][0] * 100}% \u00B1 {model_dict5[mod][1] * 100}%.')

lib.bar_plot([i for i in range(7)], [v[0] * 100 for v in model_dict5.values()], title="5-fold CV score",
             x_label='Model',
             y_label='%', LABELS=['CustomNBC', 'Kn1', 'Kn3', 'SVC(lin)', 'SVC(poly)', 'SVC', 'SVC(sigm)'])  # bar
# plot the results
print('-----------------------------------------------------------')
