# 模型加载导入方法
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset as ds
import os
import train
from src.loss import loss
from PIL import Image
import moxing as mox
import numpy as np
from mindspore import Tensor
from src.nets import net_factory


def test_data(testdataset_path):
    # 调用训练的模型
    # model = DeepLabV3(num_classes=2)
    model = net_factory.nets_map['deeplab_v3_s16'](num_classes=2)
    test_path = os.path.join(testdataset_path, "test")
    img_list = os.listdir(test_path)
    cnt = 0
    for image_name in img_list:
        cnt += 1
        image_path = os.path.join(test_path, image_name)
        # print(image_path)
        image = Image.open(image_path)

        image_dataset = np.asarray(image, dtype=np.float32)
        # 高维度转换,需要注意图像大小，根据图像大小来进行转换
        image_arr = np.transpose(image_dataset, (2, 0, 1))
        w,h = image_arr[0][0].size,image_arr[0][1].size
        image_arr = image_arr.reshape(1, 3, w, h)
        image_dataset = Tensor(image_arr)
        # 加载模型
        # # 将模型参数导入parameter的字典中
        # mox.file.copy_parallel(src_url='./output_train', dst_url='s3://southgis-train/output_train')
        #下载训练后的模型
        #关联保存模型的路径
        mox.file.copy_parallel(src_url='s3://southgis-train/output_train', dst_url='./output_train')
        param_dict = load_checkpoint("./output_train/deeplabv3-20_2500.ckpt")
        # # 将参数加载到网络中
        load_param_into_net(model, param_dict)

        # 预测的数据
        pre_output = model(image_dataset)
        pre_arr = pre_output.asnumpy().reshape(2, w, h)
        # print('pre_arr_shape:', pre_arr.shape)
        pre_arr = np.argmax(pre_arr, axis=0)
        # print('pre_arr', max(pre_arr[0]))
        # 将ndarry转换为image
        pre_img = Image.fromarray(np.uint8(pre_arr))

        #保存预测的数据
        mox.file.copy_parallel(src_url='./gis', dst_url='s3://southgis-train/gis')
        pre_img.save(os.path.join(testdataset_path + '/pre', image_name.split('.')[0] + '_mask.png'))
        # print("test=====", cnt)

    #将保存的数据进行可视化，此处可以删除
    image_path = os.path.join(testdataset_path, 'pre')
    img_list = os.listdir(image_path)
    for img_name in img_list:
        img_path = os.path.join(image_path, img_name)
        img = Image.open(img_path)
        img_arr = np.asarray(img, dtype=np.float32)
        img_arr[img_arr == 1] = 255
        new_img = Image.fromarray(np.uint8(img_arr))
        mox.file.copy_parallel(src_url='./gis', dst_url='s3://southgis-train/gis')
        new_img.save(os.path.join(testdataset_path + '/pre_test', img_name))


if __name__ == "__main__":
    # 关联测试数据路径
    mox.file.copy_parallel(src_url='s3://southgis-train/gis', dst_url='./gis')
    test_data('./gis')
