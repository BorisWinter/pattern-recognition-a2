if __name__ == "__main__":
    # Import parent folder such that we can import sibling modules
    import sys
    sys.path.append("..")

from audioop import cross
from raw_data.data_functions import load_img_data, load_num_data
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from feature_selection.fourier_transform import ft_on_num_data, ft_on_img_data

def decision_tree(data, labels, depth=5):
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=labels)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Find accuracy
    return (metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average="weighted"), y_pred)

def cross_val_decision_tree(data, labels, cv=5, depth=5):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)

    result = cross_validate(clf, data, labels, cv=cv, scoring=["accuracy", "f1_weighted"])
    test_acc = result["test_accuracy"].mean()
    test_f1 = result["test_f1_weighted"].mean()
    return (test_acc, test_f1)

def test_raw_data():
    print("Loading dataset for images...")
    img_data, labels = load_img_data()
    
    print("using decision tree classifier on images")
    accuracy = decision_tree(img_data, labels)
    print(f"Image - Decision tree accuracy: {accuracy}")
    
    
    print("Loading dataset for genes...")
    genes_data, labels = load_num_data()

    print("using decision tree classifier on genes")
    accuracy = decision_tree(genes_data, labels)
    print(f"Genes - Decision tree accuracy: {accuracy}")

def test_fourier_transform():
    print("Loading dataset for images...")
    img_data, labels = load_img_data()

    ft_image_data = ft_on_num_data(img_data, labels)
    accuracy = decision_tree(ft_image_data, labels)

    print(f"Images - Decision tree accuracy: {accuracy}")

    print("Loading dataset for genes...")
    genes_data, labels = load_num_data()

    ft_genes_data = ft_on_num_data(genes_data, labels)
    accuracy = decision_tree(ft_genes_data, labels)

    print(f"Genes - Decision tree accuracy: {accuracy}")
    

if __name__ == "__main__":
    print(" -- Raw data")
    test_raw_data()

    print(" -- Fourier Transform data")
    test_fourier_transform()