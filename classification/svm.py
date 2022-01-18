from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle,os

def svm_genes(Feat_slct,lbs_noH):
    print("svm train start")
    x_train,x_test,y_train,y_test=train_test_split(Feat_slct,lbs_noH,stratify=lbs_noH,test_size=0.2,random_state=42)

    if os.path.isfile("svm_gense.pkl"):
        svm = pickle.load(open("svm_gense.pkl", "rb"))
    else:
        svm=SVC()
        svm.fit(x_train,y_train)
        pickle.dump(svm, open("./svm_gense.pkl", "wb"))
    print("accuracy on the training subset:{:.3f}".format(svm.score(x_train,y_train)))
    print("accuracy on the test subset:{:.3f}".format(svm.score(x_test,y_test)))