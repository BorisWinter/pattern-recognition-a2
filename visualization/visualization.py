from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_conf_matrix(labels, pred_labels, title = "Confusion Matrix"):
    """
    Takes true labels and predicted labels.
    Displays a heatmap of the corresponding confusion matrix.
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
    ax.set_title(title, fontsize=40, pad=20)
    

def simple_pca_line_plot(y, title="Line Plot", ylabel="y"):
    """
    Takes a sequence of data.
    Displays a simple line plot.
    """
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(y)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Number of principal components", fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_title(title, fontsize=40, pad=20)


def plot_tsne_num(num_data_embedded, num_labels, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(15,10))
    num_labels[1].name = "Class"
    ax = sns.scatterplot(num_data_embedded[:,0], num_data_embedded[:,1], hue=num_labels[1])
    ax.tick_params(labelsize=20)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_title(title, fontsize=40, pad=20)
    plt.legend(fontsize='xx-large', title_fontsize='40')


def plot_tsne_img(img_data_embedded, img_labels, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(15,10))
    ax = sns.scatterplot(img_data_embedded[:,0], img_data_embedded[:,1], hue=img_labels)
    ax.tick_params(labelsize=20)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_title(title, fontsize=40, pad=20)
    plt.legend(fontsize='xx-large', title_fontsize='40')

