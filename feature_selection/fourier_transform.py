import numpy as np

def ft_on_img_data(img_data,labels):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))


def ft_on_num_data(num_data,labels):
    f = np.fft.fft(num_data)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))