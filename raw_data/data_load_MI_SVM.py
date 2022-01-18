import pandas as pd 
import os,cv2
import numpy as np

def load_genes_data():
    print("load data start")
    features=pd.read_csv('.././Data-211216/Data/Genes/data.csv',header=None,low_memory=False)
    labels=pd.read_csv('.././Data-211216/Data/Genes/labels.csv',header=None,low_memory=False)

    featr_nohead=features.drop(columns=0).drop(0)
    lbs_noH=labels.drop(columns=0).drop(0)
    print("load data succ")
    return featr_nohead,lbs_noH


SHAPE = (100, 100)
def extractFeaturesFromImage(image_file):
    img_org = cv2.imread(image_file,0) # greyscale
    img_prcss= img_pre_process(img_org)
    img = cv2.resize(img_prcss, SHAPE, interpolation = cv2.INTER_CUBIC)
    img = img.flatten()
    img = img / np.mean(img)
    return img
   
def img_pre_process(img):
    s = max(img.shape[0:2])#Getting the bigger side of the image
    f = np.zeros((s,s),np.uint8)#Creating a dark square with NUMPY  
    ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2#Getting the centering position
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img    #Pasting the 'image' in a centering position
    #print(f.shape)
    return f
def getImageData(directory):
    s = 1
    feature_list = list()
    label_list   = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root+d)
            for image in images:
                s += 1
                label_list.append(d)
                feature_list.append(extractFeaturesFromImage(root + d + "/" + image))
    print("data load succ")
    return np.asarray(feature_list), np.asarray(label_list)


    