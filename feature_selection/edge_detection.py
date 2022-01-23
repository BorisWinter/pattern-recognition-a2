if __name__=="__main__":
    import sys
    sys.path.append("..")

import numpy as np
import pandas as pd
from skimage.filters import roberts
from raw_data.data_functions import RESIZE_SHAPE



def prewitt_filter(img):
    # Back into 2d
    image = img.to_numpy().reshape(RESIZE_SHAPE)

    image = roberts(image)

    return pd.Series(image.flatten())


def edge_detection(img_data):
    return img_data.apply(prewitt_filter, axis=1, result_type="expand")

if __name__=="__main__":
    from raw_data.data_functions import load_img_data
    data, labels = load_img_data()
    print(edge_detection(data))
# %%
