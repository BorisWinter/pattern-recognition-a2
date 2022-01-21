from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd 

def MI_feature_select(featr_nohead,lbs_noH,thre):
    print("feature selection start")
    print("Feature drop")
    #featr_nohead=features.drop(columns=0).drop(0)

    x_y_result = mutual_info_classif(featr_nohead,lbs_noH)
    sort_indx=np.argsort(-x_y_result)
    threshold_index=0
    for i in range(len(sort_indx)):
        if x_y_result[sort_indx[i]]<thre: # threshold of MI score
            threshold_index=i
            break
    NewFeat_indx=sort_indx[:threshold_index]
    print("%d features whose MI score over threshold are selected from %d features"%(len(NewFeat_indx),featr_nohead.shape[1]))
    if 'DataFrame' in str(type(featr_nohead)):
        Feat_slct=pd.DataFrame(featr_nohead,columns = NewFeat_indx) # col feature is selected and sorted according to the MI score
        print("feature selection end")
    elif 'ndarray'in str(type(featr_nohead)):
        Feat_slct=featr_nohead[:,NewFeat_indx]
        print("feature selection end")
    return Feat_slct
    