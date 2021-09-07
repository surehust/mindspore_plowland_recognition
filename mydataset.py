import numpy as np

import mindspore.dataset as de
from PIL import Image
import os


# 自定义数据集
class DatasetGenerator:
    def __init__(self, data_root):
        self.data_root = data_root
        train_path = os.path.join(data_root, "train")
        label_root = os.path.join(data_root, "label")
        self.img_list = os.listdir(train_path)
        self.data_list = []

        for image_name in self.img_list:  # 0.jpg
            image_path = os.path.join(train_path, image_name)
            label_name = image_name.split(".")[0] + '_mask.png'
            label_path = os.path.join(label_root, label_name)
            self.data_list.append((image_path, label_path))


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, item):
        image_path = self.data_list[item][0]
        label_path = self.data_list[item][1]

        image = Image.open(image_path)  # h, w, 3
        label = Image.open(label_path)

        image_arry = np.asarray(image, dtype=np.float32)
        label_arr = np.asarray(label, dtype=np.int16)
        label_arr = label_arr[:, :, 0]
        image_arr = np.transpose(image_arry, (2, 0, 1))

        return image_arr, label_arr



