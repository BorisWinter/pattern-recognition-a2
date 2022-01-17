import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(data, k=None):
    """
    pca()
    performs PCA on the given numerical data. Outputs the k principal components.
    """

    # Normalize the data
    normalized_data = pd.DataFrame(StandardScaler().fit_transform(data))

    # Perform pca
    pca = PCA(n_components=k)
    principal_components = pd.DataFrame(pca.fit_transform(normalized_data))
    principal_components

    # Return the k principal components
    return principal_components