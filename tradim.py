import imageio
import cv2
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
#维度转换函数
def redim_fun(fileroot):
    """
    :param fileroot:图片的路径
    :return: None
    :作用：将指定路径的图片转换维度，保存在save_file_path这个路径下
    """
    crop_size =(577,375)
    img = cv2.imread(fileroot)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_new = cv2.resize(img_gray, crop_size, interpolation=cv2.INTER_CUBIC)  # 立方插值
    save_dir_path = os.path.dirname(os.path.abspath(fileroot))
    filename = fileroot.split('\\')[-1] #图片文件的名字，不包含路径前缀
    filename = filename[5:] #删除Clip_前缀
    save_file_path = os.path.join(save_dir_path,'Anyang_done',filename) #E:\....\Anyang_done\图片文件的名字
    imageio.imsave(save_file_path, img_new)  # 转换维度后的图片img_new保存到save_file_path这个路径下

if __name__ == "__main__":
    import os
    dir_path = 'E:\deep_learning\city_pre\modis_city\Anyang'
    filename_list = list()
    filesname=[]
    for root, dirs, filesname in os.walk(dir_path, topdown=False): #filename是dir_path下所有文件的名字组成的列表
        pass
    for _ in filesname: #读取每一个文件名字
        if _[-4:] == ".tif":
            filename_list.append(_) #若后缀是 .tif 则加入filename_list这个列表
    for i in range(len(filename_list)):
        redim_fun(os.path.join(dir_path,filename_list[i])) #路径+文件名，后缀是 .tif的文件都调用一次redim_fun函数
        #redim_fun函数的作用：将指定路径的图片转换维度，保存在save_file_path这个路径下
    print("done")