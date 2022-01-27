from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import StratifiedKFold,cross_validate
import pickle,os
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score
from feature_selection.SIFT import cal_vec

def svm_genes(Feat_slct,lbs_noH):
    print("svm train start")
    x_train,x_test,y_train,y_test=train_test_split(Feat_slct,lbs_noH,stratify=lbs_noH,test_size=0.2,random_state=42)
    if os.path.isfile("svm_gense1.pkl"):
        print("model exist")
        svm = pickle.load(open("svm_gense.pkl", "rb"))
    else:
        svm=SVC()
        svm.fit(x_train,y_train)
        pickle.dump(svm, open("./svm_gense.pkl", "wb"))
        print("new model saved")
    print("accuracy on the training subset:{:.3f}".format(svm.score(x_train,y_train)))
    print("accuracy on the test subset:{:.3f}".format(svm.score(x_test,y_test)))

def svm_images_gridsearch(feature_array,label_array,n_splits = 5):
    print("svm train start")
    param_grid={'C':[0.1,1,10],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','linear']}
    kf = StratifiedKFold(n_splits = n_splits, random_state = 1, shuffle=True)
    
    svm = SVC(probability=True)#C=100,kernel='rbf',gamma=0.001,
    model=GridSearchCV(svm,param_grid,cv=kf,return_train_score=True)
    grid_result=model.fit(feature_array,label_array)
    
    df = pd.DataFrame(grid_result.cv_results_)
    svm_results = pd.concat([lr_results,df])
    svm_results = svm_results.sort_values(by=['mean_test_score'], ascending=False)

    return svm_results


def svm_images_cross_val(feature_array,label_array,n_splits = 5,C=100,kernel='rbf',gamma=0.001):
    print("svm train start")
    kf = StratifiedKFold(n_splits = n_splits, random_state = 1, shuffle=True)
    
    svm = SVC(C=C,kernel=kernel,gamma=gamma,probability=True)#C=100,kernel='rbf',gamma=0.001,
    result = cross_validate(svm, feature_array,label_array, cv=kf, scoring=["accuracy", "f1_weighted"])

    acc = result["test_accuracy"].mean()
    f1 = result["test_f1_weighted"].mean()

    return acc, f1


def svm_images(X_train, X_test, y_train, y_test,C=100,kernel='rbf',gamma=0.001):
    print("svm train start")
    if os.path.isfile("svm_cat11.pkl"):
        print("model exist")
        grid_result = pickle.load(open("svm_cat.pkl", "rb"))
    else:
        svm = SVC(C=C,kernel=kernel,gamma=gamma,probability=True)#C=100,kernel='rbf',gamma=0.001,
        model=svm.fit(X_train, y_train)
        pickle.dump(model, open("./svm_cat.pkl", "wb"))
        print("new model saved")
    
    centers = np.load("./svm_centers.npy")
    #计算每张图片的特征向量
    data_vec,labels = cal_vec(X_test, y_test)
    
    pred_labels = model.predict(data_vec)
    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average="weighted")
    
    print("accuracy on the training subset:{:.3f}".format(model.score(X_train,y_train)))
    print("accuracy on the test subset:{:.3f}".format(model.score(data_vec,labels)))
    #print("success")
    
    
    return acc, f1, pred_labels