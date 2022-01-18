import pandas as pd 

def load_genes_data():
    print("load data start")
    features=pd.read_csv('.././Data-211216/Data/Genes/data.csv',header=None,low_memory=False)
    labels=pd.read_csv('.././Data-211216/Data/Genes/labels.csv',header=None,low_memory=False)

    featr_nohead=features.drop(columns=0).drop(0)
    lbs_noH=labels.drop(columns=0).drop(0)
    print("load data succ")
    return featr_nohead,lbs_noH
    