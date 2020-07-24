#!/usr/bin/env python
# organize train/test files from pickled data in subdirectories to be used with ImageDataGenerator
# to generate a flow
import os
import pickle
from sklearn.model_selection import train_test_split
from matplotlib.image import imsave, imread

pickle_file = "/media/miro/WD/jetbot_obstacle_avoidance/data_224.pckl"
test_size = 0.4

def create_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    pickle_file_name = os.path.basename(pickle_file)

    train_dir = os.path.join(os.path.dirname(pickle_file), pickle_file_name.replace('.', '_')+'_'+str(test_size), "train")
    create_dirs(train_dir)

    test_dir = os.path.join(os.path.dirname(pickle_file), pickle_file_name.replace('.', '_')+'_'+str(test_size), "test")
    create_dirs(test_dir)

    with open(pickle_file, "rb") as f:
        X, y, name_to_idx, idx_to_name = pickle.load(f)

    print(name_to_idx, idx_to_name)

    for class_name in name_to_idx.keys():
        create_dirs(os.path.join(train_dir, class_name))
        create_dirs(os.path.join(test_dir, class_name))

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

    # print(type(X_train), X_train.min(), X_train.max(), X_train.shape, X_train[0].dtype)
    # print(type(y_train), y_train.min(), y_train.max(), y_train.shape, y_train[0].dtype)

    # export pickled images of train and test split in corresponding directories
    for i in range(len(X_train)):
        train_file = os.path.join(train_dir, idx_to_name[y_train[i]], str(i)+'.jpg')
        imsave(train_file, X_train[i])
        print(train_file)

    for i in range(len(X_test)):
        test_file = os.path.join(test_dir, idx_to_name[y_test[i]], str(i)+'.jpg')
        imsave(test_file, X_test[i])
        print(test_file)


