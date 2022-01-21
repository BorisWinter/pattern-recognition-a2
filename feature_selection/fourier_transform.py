import numpy as np
import pandas as pd
def ft_on_img_data(img_data):
    new_df = pd.DataFrame(index=np.arange(len(img_data)), columns=np.arange(len(img_data.iloc[0])))
    for idx, img in img_data.iterrows():
        print(f"processing image #{idx+1} out of {len(img_data)}", end="\r")
        f = np.fft.fft(img)
        fshift = np.fft.fftshift(f)
        new_df.append(pd.Series(20*np.log(np.abs(fshift))), ignore_index=True)
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

    ft_on_img_data(img_data)