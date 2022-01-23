import pandas as pd
import os
import cv2

RESIZE_SHAPE = (150,150)

def load_img_data():
    # Import the image data and convert to vectors
    img_vectors = []
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'BigCats')
    classes = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
    img_labels = []

    for img_class in classes:
        class_path = os.path.join(path, img_class)

        for file in os.listdir(class_path):
            img_path = class_path + "/" + file
            # print(img_path)
                
            # Read image
            image = cv2.imread(img_path, 0)

            # Resize image so all have the same dimensions
            resized = cv2.resize(image, dsize=RESIZE_SHAPE)

            # Flatten the image
            flat = resized.flatten()

            # Add the image vector to the data set
            # img_vectors = np.vstack([img_vectors, flat])
            img_vectors.append(flat)

            # Add the image class to the labels
            img_labels.append(img_class)

    img_data = pd.DataFrame(img_vectors)

    return img_data, img_labels

def load_num_data():
    # Import numerical data
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'Genes')
    data = pd.read_csv(os.path.join(path, 'data.csv'), header=None, low_memory=False)
    labels = pd.read_csv(os.path.join(path, 'labels.csv'), header=None, low_memory=False)
    
    return data.drop(columns=0).drop(0), labels.drop(columns=0).drop(0)

if __name__=="__main__":
    print(load_num_data())