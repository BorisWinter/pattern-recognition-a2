from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics

def naive_bayes(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels.to_numpy().ravel(), test_size=0.2, random_state=42)
    gnb = GaussianNB()

    gnb = gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    return (metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average="weighted"), y_pred)


def naive_bayes_cv(data, labels, cv=10):
    gnb = GaussianNB()

    result = cross_validate(gnb, data, labels.to_numpy().ravel(), cv=cv, scoring=["accuracy", "f1_weighted"])
    test_acc = result["test_accuracy"].mean()
    test_f1 = result["test_f1_weighted"].mean()

    return (test_acc, test_f1)