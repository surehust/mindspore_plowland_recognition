

import cv2
import numpy as np
import os
import moxing as mox


# 数据预处理
def img_preprocess(prepro_root):
    out_path = os.path.join(prepro_root, "labels")
    out_dir = os.listdir(out_path)
    for image_name in out_dir:
        image_path = os.path.join(out_path, image_name)
        img = cv2.imread(image_path)
        img[img == 255] = 1
        mox.file.copy_parallel(src_url='./gis', dst_url='s3://southgis-train/gis')
        cv2.imwrite('./gis/label/{}'.format(image_name), img)


if __name__ == "__main__":
    # 关联数据路径
    mox.file.copy_parallel(src_url='s3://southgis-train/gis', dst_url='./gis')
    img_preprocess('./gis')





