import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

paths=["D:\PycharmProjects\RoomFormer-main\data_preprocess\stru3d\Structured3D_panorama\Structured3D_panorama_00\Structured3D\scene_00000\density.jpg"]

save_paths=["D:\PycharmProjects\RoomFormer-main\data_preprocess\stru3d\Structured3D_panorama\Structured3D_panorama_00\Structured3D\scene_00000/1.jpg"]

# 直方图规定化
def histogram_equalization(image,mu=128,sigma=50):
    # 计算图像的直方图
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    # print(hist)

    # 定义高斯分布的均值和标准差
    mu = mu
    sigma = sigma

    # 计算累积分布函数
    cdf = hist.cumsum()/hist.sum()
    # print(cdf)

    # 生成高斯分布累计分布函数
    x= np.linspace(0,256,256)
    # print(x)
    cdf_normalized = stats.norm(mu,sigma).cdf(x)
    # print(cdf_normalized)

    p = np.zeros(256)
    j = 0
    for i in range(256):
        while (cdf[i] > cdf_normalized[j] and j<255):
            j += 1
        p[i] = j

    # 使用累积分布函数进行直方图规定化
    equalized_image = np.interp(image.flatten(), bins[:-1], p)

    # 将图像重新恢复形状
    equalized_image = equalized_image.reshape(image.shape)

    return equalized_image

# 生成对比图
def comparision():
    for i in range(len(paths)):
        image = cv2.imread(paths[i], 0)

        #直方图均衡化
        equalized_image = cv2.equalizeHist(image)
        # #直方图规定化
        # equalized_image = histogram_equalization(image,mu=128,sigma=50)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5)
        plt.title('Original Histogram')

        plt.subplot(2, 2, 3)
        plt.imshow(equalized_image, cmap='gray')
        plt.title('Equalized Image')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='green', alpha=0.5)
        plt.title('Equalized Histogram')

        plt.tight_layout()
        plt.savefig(
            "D:\PycharmProjects\RoomFormer-main\data_preprocess\stru3d\Structured3D_panorama\Structured3D_panorama_00\Structured3D\scene_00000/1.jpg")
        plt.show()


comparision()

