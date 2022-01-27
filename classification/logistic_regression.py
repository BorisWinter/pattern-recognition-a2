# Imports
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, cross_validate

def logistic_regression(train_data, train_labels, test_data, test_labels, solver, c_value):
    """
    Performs logistic regression on the given data.
    """

    lr = LogisticRegression(random_state=0, max_iter=1000)
    lr.fit(train_data, train_labels)
    pred_labels = lr.predict(test_data)

    acc = accuracy_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels, average="weighted")

    return acc, f1, pred_labels


def logistic_regression_cross_validation(data, labels, solver, c_value):
    """
    Evaluates logistic regression using 10-fold cross validation on the given data.
    """

    # Cross validation settings
    kf = StratifiedKFold(n_splits = 10, random_state = 1, shuffle=True)

    # Create the k-NN classifier
    lrcv = LogisticRegression(random_state=0, max_iter=1000, solver=solver, C=c_value)

    result = cross_validate(lrcv, data, labels, cv=kf, scoring=["accuracy", "f1_weighted"])

    acc = result["test_accuracy"].mean()
    f1 = result["test_f1_weighted"].mean()

    return acc, f1


def logistic_regression_gridsearch(data, labels, solvers, c_values, n_splits = 10):
    """
    logistic_regression_gridsearch()
     - Performs gridsearch for logistic regression on the given data.
    """

    # Create dataframe for storage
    lr_results = pd.DataFrame([], columns = ["param_solver", "param_C", "mean_train_score", "mean_test_score"])

    # Cross validation settings
    kf = StratifiedKFold(n_splits = n_splits, random_state = 1, shuffle=True)

    # Define the model
    lr_model = LogisticRegression(random_state=0, max_iter=1000)
    parameters = {'C':[i for i in c_values], 'solver':[s for s in solvers]}

    clf = GridSearchCV(lr_model, parameters, cv=kf, return_train_score=True)
    clf.fit(data, labels)

    df = pd.DataFrame(clf.cv_results_)
    lr_results = pd.concat([lr_results,df])
    lr_results = lr_results.sort_values(by=['mean_test_score'], ascending=False)

    return lr_results