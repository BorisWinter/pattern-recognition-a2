
import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split


# Better Partition coefficient: https://reader.elsevier.com/reader/sd/pii/S1877705811055433?token=28A39E088227C2B0B521D4FDF4EE7D307FB92616A5F6AC7037B1271284C05C2A57AA5972A4CCF1BDB0221454FF7A2473&originRegion=eu-west-1&originCreation=20220122135233
# Maybe needed because normal PC favours higher cluster numbers

def fuzzy_c_means(data, labels, cluster_number=5, m=2):

    fcm = FCM(n_clusters=cluster_number)
    fcm.m = m
    fcm.fit(data.to_numpy(dtype=np.float32))

    # outputs
    fcm_centers = fcm.centers
    fcm_labels = fcm.predict(data.to_numpy(dtype=np.float32))

    # https://reader.elsevier.com/reader/sd/pii/0098300484900207?token=1ABCC157BD96B91E805094F31FEC2355157FF8119DC1EF79C3244BC65F059A569F1D57E7E032D58FFC133536C37D0727&originRegion=eu-west-1&originCreation=20220122135012
    # See equation 12a in this paper on FCM by Bezdek
    print(f"{fcm.partition_coefficient=}")
    print(f"{fcm.partition_entropy_coefficient=}")

    sil_score = silhouette_score(data, fcm_labels)
    print(f"{sil_score=}")
    rand_score = adjusted_rand_score(labels.to_numpy().flatten(), fcm_labels)
    print(f"{rand_score=}")
    mi_score = adjusted_mutual_info_score(labels.to_numpy().flatten(), fcm_labels)
    print(f"{mi_score=}")
    return sil_score, rand_score, mi_score

def fuzzy_c_means_cv(data, labels, cluster_number=5):
    pass


if __name__=="__main__":
    import sys
    sys.path.append("..")
    from raw_data.data_functions import load_num_data
    from tqdm import tqdm
    data, labels = load_num_data()

    cluster_numbers = [3,4,5,6,7,8,9,10]
    m = [1,2,3,4]

    pc = []
    pec = []
    for cluster in tqdm(cluster_numbers):
        for num in m:
            print(f"cluster number: {cluster}")
            result = fuzzy_c_means(data, labels, cluster, m)
            pc.append(result[0])
            pec.append(result[0])
    
    for idx, _ in enumerate(pc):
        print(f"{pc[idx] - pec[idx]}")
