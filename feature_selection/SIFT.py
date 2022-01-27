import os
import numpy as np
import cv2

def calcSiftFeature(img):
    sift = cv2.SIFT_create()
    keypoints, features = sift.detectAndCompute(img, None)
    return features

def learnVocabulary(features):
    wordCnt = 50
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(features, wordCnt, None,criteria, 20, flags)
    return centers

def calcFeatVec(features, centers):
    featVec = np.zeros((1, 50))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (50, 1)) - centers
        sqSum = (diffMat**2).sum(axis=1)
        dist = sqSum**0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]
        featVec[0][idx] += 1
    return featVec

def build_center(img_data):
    features = np.float32([]).reshape(0, 128)
    for im in img_data:
        img_f = calcSiftFeature(im)
        features = np.append(features, img_f, axis=0)
    centers = learnVocabulary(features)
    filename = "./svm_centers.npy"
    np.save(filename, centers)
    print('Vocabulary saved:',centers.shape)

def cal_vec(img_data, img_labels):
    centers = np.load("./svm_centers.npy")
    data_vec = np.float32([]).reshape(0, 50)
    labels = np.float32([])
    for i in range(len(img_data)): 
        img_f = calcSiftFeature(img_data[i])
        img_vec = calcFeatVec(img_f, centers)
        data_vec = np.append(data_vec,img_vec,axis=0)
        labels = np.append(labels,img_labels[i])
            
    print('data_vec:',data_vec.shape)
    print('image features extration done!')
    return data_vec,labels