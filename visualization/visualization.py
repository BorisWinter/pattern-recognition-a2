from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_conf_matrix(labels, pred_labels):
    """
    Takes true labels and predicted labels.
    Returns a heatmap of the corresponding confusion matrix.
    """

    classes = np.unique(np.append(labels, pred_labels))
    conf_matrix = confusion_matrix(labels, pred_labels)
    return_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10,10))
    sns. set(font_scale=1.8)
    ax = sns.heatmap(return_matrix, annot=True, fmt=".2%", linewidth=.5, cbar=False, xticklabels=classes, yticklabels=classes)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Predicted class", fontsize=30)
    ax.set_ylabel("True class", fontsize=30)
    ax.set_title("Confusion matrix", fontsize=40)

    return return_matrix