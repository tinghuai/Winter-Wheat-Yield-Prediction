import imageio
import cv2
import numpy as np

from osgeo import gdal
def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset  # 关闭对象dataset，释放内存
    # return im_width, im_height, im_proj, im_geotrans, im_data,im_bands
    return [im_width, im_height]
crop_size = (577, 375)
load_npy = np.load("E:\\deep_learning\\city_pre\\npdata\\modis_data_np_0.npy")
print(load_npy.shape)


