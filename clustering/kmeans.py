if __name__ == "__main__":
    # Import parent folder such that we can import sibling modules
    import sys
    sys.path.append("..")

import os,cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, k_means
from imutils import build_montages
import matplotlib.image as imgplt
from feature_selection.MI import MI_feature_select
from raw_data.data_load_MI_SVM import getImageData

from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

def kmeans_train(feature_array, label_array, clusters=5):
    
    #feature_array=MI_feature_select(feature_array, label_array)
  
    clt = KMeans(n_clusters=clusters)
    clt.fit(feature_array)
    #labelIDs = np.unique(clt.labels_)

    y_test_pre=clt.predict(feature_array)  #,y=y_test

    sil_score = silhouette_score(feature_array, y_test_pre)
    mi_score = adjusted_mutual_info_score(label_array, y_test_pre)
    rand_score = adjusted_rand_score(label_array, y_test_pre)

    return sil_score, rand_score, mi_score

    # for labelID in labelIDs:
    #     print("----------------k=",labelID,"----------------")
    #     idxs = np.where(clt.labels_ == labelID)[0]
    #     idxs = np.random.choice(idxs, size=min(25, len(idxs)),
    #         replace=False)
    #     #print(idxs)
    #     print(y_train[idxs])
    #     show_box = []
    #     for i in idxs:
    #         #image = cv2.imread(image_path[i])
    #         image1=np.reshape(X_train[i],(100,100))
    #         image = cv2.resize(image1, (100,100)) # for visualize
    #         image=np.array(image,dtype=np.uint8)
    #         image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #         show_box.append(image)
    #     montage = build_montages(show_box, (100,100), (5, 5))[0]
    #     title = "k= {}".format(labelID)
    #     cv2.imwrite("k= %d.jpg"%labelID, montage)

if __name__=="__main__":
    import sys
    sys.path.append("..")
    from raw_data.data_functions import load_img_data
    from tqdm import tqdm
    data, labels = load_img_data()

    cluster_numbers = [3,4,5,6,7,8,9,10]

    scores = []
    for cluster in tqdm(cluster_numbers):
        print(f"cluster number: {cluster}")
        scores.append(kmeans_train(data, labels, cluster)[0])
    print(scores)
    print(max(scores))