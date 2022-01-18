from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle,os

def svm_genes(Feat_slct,lbs_noH):
    print("svm train start")
    x_train,x_test,y_train,y_test=train_test_split(Feat_slct,lbs_noH,stratify=lbs_noH,test_size=0.2,random_state=42)
    if os.path.isfile("svm_gense.pkl"):
        print("model exist")
        svm = pickle.load(open("svm_gense.pkl", "rb"))
    else:
        svm=SVC()
        svm.fit(x_train,y_train)
        pickle.dump(svm, open("./svm_gense.pkl", "wb"))
        print("new model saved")
    print("accuracy on the training subset:{:.3f}".format(svm.score(x_train,y_train)))
    print("accuracy on the test subset:{:.3f}".format(svm.score(x_test,y_test)))

def svm_images(feature_array,label_array):
    print("svm train start")
    X_train, X_test, y_train, y_test = train_test_split(feature_array,label_array,test_size=0.2,random_state=77,stratify=label_array)
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}

    if os.path.isfile("svm_cat.pkl"):
        print("model exist")
        grid_result = pickle.load(open("svm_cat.pkl", "rb"))
    else:
        svm = SVC(probability=True)#C=100,kernel='rbf',gamma=0.001,
        model=GridSearchCV(svm,param_grid,n_jobs=-1)
        grid_result=model.fit(X_train, y_train)
        print("Best model using %s" % (model.best_params_))
        pickle.dump(grid_result, open("./svm_cat.pkl", "wb"))
        print("new model saved")
        
    print("accuracy on the training subset:{:.3f}".format(grid_result.score(X_train,y_train)))
    print("accuracy on the test subset:{:.3f}".format(grid_result.score(X_test,y_test)))
    print("success")