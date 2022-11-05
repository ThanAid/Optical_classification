import pandas as pd
import lib  # import lib.py that contains functions created for this exercise
import warnings

warnings.filterwarnings("ignore")  # Ignore Warnings

# Step 1 -- Read data and assign values.
test_data = pd.read_csv('test.txt', sep=' ', header=None)
train_data = pd.read_csv('train.txt', sep=' ', header=None)
test_data = test_data.drop([257], axis=1)  # last column was containing NaN values
train_data = train_data.drop([257], axis=1)  # last column was containing NaN values

y_test = test_data[0].to_numpy()
X_test = test_data.drop([0], axis=1).to_numpy()
y_train = train_data[0].to_numpy()
X_train = train_data.drop([0], axis=1).to_numpy()


# Step 19 -- Building and evaluating a NN
print('\n-------------------------Custom NN model---------------------------------')
nn_model = lib.PytorchNNModel(batch_size=32, layers=[60, 60], n_classes=10, n_features=X_train.shape[1], epochs=80,
                              learning_rate=0.1)
nn_model.fit(X_train, y_train, test_size=0.2, printing=False)  # fit model with 80-20 split data
print(f'Accuracy score on train data (80 - 20 split) {100 * nn_model.accuracy}%.')
print(f'Actual model Score on Test data (80 - 20 split): {nn_model.score(X_test, y_test) * 100}%.')
voting_score5, std5 = lib.evaluate_classifier(nn_model, X_train, y_train)
print(f'Model has 5 fold CV {voting_score5 * 100}% \u00B1 {std5 * 100}%.')

nn_model = lib.PytorchNNModel(batch_size=32, layers=[60, 60], n_classes=10, n_features=X_train.shape[1], epochs=80,
                              learning_rate=0.1)
nn_model.fit(X_train, y_train)  # fit model with all the train data
print(f'\nModel Score trained with whole train data: {nn_model.score(X_test, y_test) * 100}%.')
voting_score5, std5 = lib.evaluate_classifier(nn_model, X_train, y_train)
print(f'Model has 5 fold CV {voting_score5 * 100}% \u00B1 {std5 * 100}%.')

print('\n--------------------------------------------------------------------------')
