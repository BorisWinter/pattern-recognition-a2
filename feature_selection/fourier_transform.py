import numpy as np
import pandas as pd

def fourier_transform_on_single_image(image):
    f = np.fft.fft(image)
    fshift = np.fft.fftshift(f)
    return 10*np.log(np.abs(fshift))

def ft_on_img_data(img_data):
    new_df = img_data.apply(fourier_transform_on_single_image, axis=1, result_type="expand")
    return new_df


def ft_on_num_data(num_data):
    f = np.fft.fft(num_data)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

if __name__=="__main__":
    import sys
    sys.path.append("..")
    from raw_data.data_functions import load_img_data
    img_data, img_labels = load_img_data()

    print(img_data)
    print(ft_on_img_data(img_data))