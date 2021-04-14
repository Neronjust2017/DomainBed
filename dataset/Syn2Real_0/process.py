import numpy as np
import os,shutil

if __name__ == '__main__':

    # # processing test data
    # image_list_file = 'image_list.txt'
    # with open(image_list_file, 'r') as f:
    #     image_list = f.readlines()
    #
    # # class:
    # # aeroplane bicycle bus car horse knife motorcycle person plant skateboard train truck (0-11)
    # test_dir = 'test'
    # test_raw_dir = 'test_raw'
    # classes = ['aeroplane', 'bicycle', 'bus','car', 'horse', 'knife',
    #            'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
    # for cla in classes:
    #     os.mkdir(os.path.join(test_dir, cla))
    # image_lists = [[] for i in range(12)]
    #
    # for image in image_list:
    #     file_name, cla = image.replace('\n', '').split(' ')
    #     cla = int(cla)
    #     image_lists[cla].append(file_name)
    #     shutil.move(os.path.join(test_raw_dir, file_name), os.path.join(test_dir, classes[cla]))
    # print('done.')

    # moving part of data
    # class:
    # aeroplane bicycle bus car horse knife motorcycle person plant skateboard train truck (0-11)
    classes = ['aeroplane', 'bicycle', 'bus','car', 'horse', 'knife',
               'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
    ratio = 0.2

    domains = ['Real1', 'Real2', 'Synthetic']

    for dom in domains:
        for cla in classes:
            os.makedirs(os.path.join('Syn2Real', dom, cla))
            image_lists = os.listdir(os.path.join(dom, cla))

            # random shuffle
            index = np.random.permutation(len(image_lists))

            for i in range(int(ratio*len(index))):
                shutil.copy(os.path.join(dom, cla, image_lists[index[i]]), os.path.join('Syn2Real', dom, cla))

    print('done.')






