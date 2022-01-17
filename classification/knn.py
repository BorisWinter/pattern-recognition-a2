# Imports
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def knn(train_data, train_labels, test_data, k):
    """
    Performs kNN on the given data.
    """

    neighbours = KNeighborsClassifier(n_neighbors=k)
    neighbours.fit(train_data, train_labels)

    return neighbours.predict(test_data)


def knn_gridsearch(data, labels, k_range, n_splits = 10):
    """
    knn_gridsearch()
     - Performs gridsearch for kNN on the given data.
    """

    # Create dataframe for storage
    knn_results = pd.DataFrame([], columns = ["param_n_neighbors", "mean_train_score", "mean_test_score"])

    # Cross validation settings
    kf = KFold(n_splits = n_splits, random_state = 1, shuffle=True)

    # Define the model
    knn_model = KNeighborsClassifier()
    parameters = {'n_neighbors':[i for i in k_range]}

    clf = GridSearchCV(knn_model, parameters, cv=kf, return_train_score=True)
    clf.fit(data, labels)

    df = pd.DataFrame(clf.cv_results_)
    knn_results = pd.concat([knn_results,df])
    knn_results = knn_results.sort_values(by=['mean_test_score'], ascending=False)

    return knn_results