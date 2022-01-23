
import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score


# Better Partition coefficient: https://reader.elsevier.com/reader/sd/pii/S1877705811055433?token=28A39E088227C2B0B521D4FDF4EE7D307FB92616A5F6AC7037B1271284C05C2A57AA5972A4CCF1BDB0221454FF7A2473&originRegion=eu-west-1&originCreation=20220122135233
# Maybe needed because normal PC favours higher cluster numbers

def fuzzy_c_means(data, labels, cluster_number=5):
    y = labels.to_numpy().flatten()
    X = data.to_numpy(dtype=np.float32)
    fcm = FCM(n_clusters=cluster_number)
    fcm.fit(X)

    # outputs
    fcm_centers = fcm.centers
    fcm_labels = fcm.predict(X)

    # https://reader.elsevier.com/reader/sd/pii/0098300484900207?token=1ABCC157BD96B91E805094F31FEC2355157FF8119DC1EF79C3244BC65F059A569F1D57E7E032D58FFC133536C37D0727&originRegion=eu-west-1&originCreation=20220122135012
    # See equation 12a in this paper on FCM by Bezdek
    print(f"{fcm.partition_coefficient=}")
    print(f"{fcm.partition_entropy_coefficient=}")

    sil_score = silhouette_score(X, fcm_labels)
    print(f"{sil_score=}")
    rand_score = adjusted_rand_score(y, fcm_labels)
    print(f"{rand_score=}")
    mi_score = adjusted_mutual_info_score(y, fcm_labels)
    print(f"{mi_score=}")
    return sil_score, rand_score, mi_score

if __name__=="__main__":
    import sys
    sys.path.append("..")
    from raw_data.data_functions import load_num_data
    from tqdm import tqdm
    data, labels = load_num_data()

    cluster_numbers = [3,4,5,6,7,8,9,10]

    scores = []
    for cluster in tqdm(cluster_numbers):
        print(f"cluster number: {cluster}")
        scores.append(fuzzy_c_means(data, labels, cluster)[0])
    print(scores)
    print(max(scores))
