import imageio
import cv2

# crop_size = (1255, 1164)
# img = cv2.imread('D:\Datasets\done11\MOD11A2\Clip_MOD11A2.A2002273.LST_Day_1km.tif')
# print(img.shape)
# img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#
# print(img_gray.shape)
# img_new = cv2.resize(img_gray, crop_size, interpolation = cv2.INTER_CUBIC)#立方插值
# print(img_new.shape)
# #CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法..
# #还可以改为cv2.INTER_LINEAR（双线性插值），
# imageio.imsave('D:\Datasets\done11\MOD11A2_done\MOD11A2.A2002273.LST_Day_1km.tif', img_new) #保存到本地

#维度转换函数
def redim_fun(fileroot):
    """
    :param fileroot:图片的路径
    :return: None
    :作用：将指定路径的图片转换维度，保存在save_file_path这个路径下
    """
    crop_size =(1255,1164)
    img = cv2.imread(fileroot)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_new = cv2.resize(img_gray, crop_size, interpolation=cv2.INTER_CUBIC)  # 立方插值
    save_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(fileroot))) #D:\Datasets\done11
    filename = fileroot.split('\\')[-1] #图片文件的名字，不包含D:\Datasets\done11\MOD11A2这个路径前缀
    filename = filename[5:] #Clip_MOD11A2.A2002273.LST_Day_1km.tif-->MOD11A2.A2002273.LST_Day_1km.tif
    save_file_path = os.path.join(save_dir_path,'MOD11A2_done',filename) #D:\Datasets\done11\MOD11A2_done\图片文件的名字
    imageio.imsave(save_file_path, img_new)  # 转换维度后的图片img_new保存到save_file_path这个路径下

if __name__ == "__main__":
    import os
    dir_path = 'D:\Datasets\done11\MOD11A2'
    filename_list = list()
    for root, dirs, filesname in os.walk(dir_path, topdown=False): #filename是dir_path下所有文件的名字组成的列表
        pass
    for _ in filesname: #读取每一个文件名字
        if _[-4:] == ".tif":
            filename_list.append(_) #若后缀是 .tif 则加入filename_list这个列表
    for i in range(len(filename_list)):
        redim_fun(os.path.join(dir_path,filename_list[i])) #路径+文件名，后缀是 .tif的文件都调用一次redim_fun函数
        #redim_fun函数的作用：将指定路径的图片转换维度，保存在save_file_path这个路径下
    print("done")
