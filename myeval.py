import numpy as np
import os
from PIL import Image
import moxing as mox


# 对测试的数据进行评估
def evaluate_standard(evaluate_root):
    gt_path = os.path.join(evaluate_root,'test_label')
    pre_path = os.path.join(evaluate_root,'pre')
    img_list = os.listdir(pre_path)
    cnt, sum = 0, 0
    for img_name in img_list:
        gt_img_path = os.path.join(gt_path,img_name)
        pre_img_path = os.path.join(pre_path,img_name)
        gt_image = Image.open(gt_img_path)
        pre_image = Image.open(pre_img_path)
        gt_array = np.asarray(gt_image, dtype=np.int16)
        gt_arr = gt_array[:, :, 0]
        pre_arr = np.asarray(pre_image, dtype=np.int16)

        matrix = generate_matrix(
            gt_arr,pre_arr
        )
        fwiou = Frequency_Weighted_Intersection_over_Union(matrix)
        print('{}_fwiou==='.format(img_name),fwiou)
        cnt += 1
        sum += fwiou
    evaluate_FWIou = sum / cnt
    return evaluate_FWIou


def generate_matrix(gt_image, pre_image, num_class=2):
    mask = (gt_image >= 0) & (gt_image < num_class)

    lab = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(lab, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)  # 2 * 2(for pascal)
    return confusion_matrix


#FWIoU计算
def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU



if __name__ == '__main__':
    mox.file.copy_parallel(src_url='s3://southgis-train/gis', dst_url='./gis')
    fwIOU = evaluate_standard('./gis')
    print('final_evaluete_fwiou:',fwIOU)

















