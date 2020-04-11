import os
from path_const import *

def dataset_building_and_update(name_path, path):
    '''
    Create and update the directories in the dataset according to
    the names.txt in the info directory
    '''
    with open(name_path,'r') as f:
        names = f.read().split(',')
    # print(names)
    old_names = set(os.listdir(path))
    removed_names = old_names - set(names)
    # remove the old directories
    for root, dirs, files in os.walk(path):
        if root.split('/')[-1] in removed_names:
            for file in files:
                os.remove(os.path.join(root, file))
            os.rmdir(root)
    # create the newly-added directories
    for name in names:
        if not os.path.exists(os.path.join(path, name)):
            os.mkdir(os.path.join(path, name))


if __name__ == "__main__":
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)
    if not os.path.exists(test_data_path):
        os.mkdir(test_data_path)
    dataset_building_and_update(name_path, train_data_path)
    dataset_building_and_update(name_path, test_data_path)
    if not os.path.exists(stranger_data_path):
        os.mkdir(stranger_data_path)