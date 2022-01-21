import numpy as np

def ft_on_img_data(img_data):
    print(len(img_data))
    f = np.fft.fft2(img_data)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))


def ft_on_num_data(num_data):
    f = np.fft.fft(num_data)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

if __name__=="__main__":
    import sys
    sys.path.append("..")
    from raw_data.data_functions import load_img_data
    img_data, img_labels = load_img_data()

    ft_on_img_data(img_data)