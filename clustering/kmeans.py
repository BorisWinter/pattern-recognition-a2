import os,cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imutils import build_montages
import matplotlib.image as imgplt
from sklearn import metrics
from feature_selection.MI import MI_feature_select
from raw_data.data_load_MI_SVM import getImageData

def kmeans_train(feature_array, label_array):
    
    #feature_array=MI_feature_select(feature_array, label_array)
    
    X_train, X_test, y_train, y_test = train_test_split(feature_array,label_array,test_size=0.2,random_state=77,stratify=label_array)
    
    clt = KMeans(n_clusters=5)
    clt.fit(X_train,y=y_train)
    labelIDs = np.unique(clt.labels_)
    
    result=clt.score(X_test,y=y_test)  # 
    print("score:",result)
    y_test_pre=clt.predict(X_test)  #,y=y_test
    MI_evaluate_score_of_test=metrics.mutual_info_score(y_test, y_test_pre)
    print("MI_evaluate_score_of_test:",MI_evaluate_score_of_test)
    for labelID in labelIDs:
        print("----------------k=",labelID,"----------------")
        idxs = np.where(clt.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)),
            replace=False)
        #print(idxs)
        print(y_train[idxs])
        show_box = []
        for i in idxs:
            #image = cv2.imread(image_path[i])
            image1=np.reshape(X_train[i],(100,100))
            image = cv2.resize(image1, (100,100)) # for visualize
            image=np.array(image,dtype=np.uint8)
            image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            show_box.append(image)
        montage = build_montages(show_box, (100,100), (5, 5))[0]
        title = "k= {}".format(labelID)
        cv2.imwrite("k= %d.jpg"%labelID, montage)