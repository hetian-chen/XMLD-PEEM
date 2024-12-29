import numpy as np
import tifffile
import glob
import os
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
import sys

os.chdir(sys.path[0])

fold_name = './Scan/'
file_name = 'Scan003'  # 完整文件名
roi = (120, 300, 80, 80)  # 特征位置矩形选框 (x, y, w, h)


def marquee_correlation(image0, image1):
    """
    计算两个图像之间的漂移 (dx, dy)。

    Parameters:
        image0 (numpy.ndarray): 参考图像。
        image1 (numpy.ndarray): 目标图像。

    Returns:
        dx (float): 水平漂移。
        dy (float): 垂直漂移。
    """
    assert image0.shape == image1.shape, "Images must have the same dimensions"
    np_height, np_width = image0.shape

    # 去除均值
    image0 = image0 - np.mean(image0)
    image1 = image1 - np.mean(image1)

    # 计算 2D FFT
    fft_size = (2 * np_height, 2 * np_width)
    fft0 = fft2(image0, s=fft_size)
    fft1 = fft2(image1, s=fft_size)

    # 计算频域的互相关
    cross_corr = fft0 * np.conj(fft1)
    corr_image = fftshift(ifft2(cross_corr).real)

    # 找到互相关矩阵中的峰值
    max_loc = np.unravel_index(np.argmax(corr_image), corr_image.shape)
    max_row, max_col = max_loc

    # 计算漂移
    center_row = corr_image.shape[0] // 2
    center_col = corr_image.shape[1] // 2
    dy = max_row - center_row
    dx = max_col - center_col

    return dx, dy


def average_ima(fold_name, file_name, ref_file):
    image_files = glob.glob(fold_name + file_name)
    images = [tifffile.imread(img_file).astype(np.float32) / 65535 for img_file in image_files]  # 归一化到 [0, 1]

    if not images:
        print("No images found, please check the path and filename.")
        return

    # ROI 参数
    x, y, w, h = roi

    # 加载参考图像并提取模板
    ref_im = glob.glob(ref_file)
    reference_image = tifffile.imread(ref_im[0]).astype(np.float32) / 65535  # 归一化
    template = reference_image[y:y+h, x:x+w]

    # 创建文件夹保存标记图像
    os.makedirs(fold_name + 'marked_images' + file_name[-7:-4], exist_ok=True)

    for idx, image in enumerate(images):
        current_roi = image[y:y+h, x:x+w]

        # 计算漂移
        dx, dy = marquee_correlation(template, current_roi)

        # 平移图像
        shifted_image = np.roll(np.roll(image, int(dy), axis=0), int(dx), axis=1)

        # 保存平移后的图像（高精度 TIFF）
        os.makedirs(fold_name + 'shifted_images' + file_name[-7:-4], exist_ok=True)
        tifffile.imwrite(fold_name + 'shifted_images' + file_name[-7:-4] + f'/shifted_image_{idx+1}.tif',
                         (shifted_image * 65535).astype(np.uint16))  # 保存为 uint16

        # 可视化标记
        image_bgr = np.dstack([4*image] * 3)  # 转为伪彩色
        # 绘制矩形边框（仅用于可视化）
        # 蓝色矩形（原始 ROI 区域）
        image_bgr[y, x:x+w, :] = [1, 0, 0]  # 上边
        image_bgr[y+h-1, x:x+w, :] = [1, 0, 0]  # 下边
        image_bgr[y:y+h, x, :] = [1, 0, 0]  # 左边
        image_bgr[y:y+h, x+w-1, :] = [1, 0, 0]  # 右边

        # 红色矩形（匹配区域）
        matched_top_left = (x - dx, y - dy)
        matched_bottom_right = (matched_top_left[0] + w, matched_top_left[1] + h)
        image_bgr[matched_top_left[1], matched_top_left[0]:matched_bottom_right[0], :] = [0, 0, 1]  # 上边
        image_bgr[matched_bottom_right[1]-1, matched_top_left[0]:matched_bottom_right[0], :] = [0, 0, 1]  # 下边
        image_bgr[matched_top_left[1]:matched_bottom_right[1], matched_top_left[0], :] = [0, 0, 1]  # 左边
        image_bgr[matched_top_left[1]:matched_bottom_right[1], matched_bottom_right[0]-1, :] = [0, 0, 1]  # 右边

        # 可视化并保存
        image_bgr_display = np.clip(image_bgr / np.max(image_bgr), 0, 1)
        plt.figure(figsize=(8, 6))
        plt.imshow(image_bgr_display)
        plt.axis('off')
        output_path = fold_name + 'marked_images' + file_name[-7:-4] + f'/marked_image_{idx+1}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()


fold_name = './Scan/'
ref_file = fold_name + file_name + '_01_001.tif'
file_name1 = file_name + '_*_001.tif'
file_name2 = file_name + '_*_002.tif'

average_ima(fold_name, file_name1, ref_file)
average_ima(fold_name, file_name2, ref_file)


def read_ima(file_name3):
    image_files = glob.glob(file_name3)
    # print(file_name3)
    images = [tifffile.imread(img_file) for img_file in image_files]
    return images

file_name1 = '001/shifted_image_*.tif'
file_name2 = '002/shifted_image_*.tif'

im1 = read_ima(fold_name +'shifted_images' + file_name1)
im2 = read_ima(fold_name +'shifted_images' + file_name2)

accumulator = np.zeros_like(im1[0], dtype=np.float32)

num = len(im1)
for i in range(num):
    dv = np.divide(im2[i], im1[i], out=np.zeros_like(im2[i], dtype=float), where=im1[i]!=0)
    accumulator += dv

average_image = accumulator / float(num)

# 步骤5：保存结果图像（带颜色条）
plt.figure(figsize=(8, 6))
plt.imshow(average_image, cmap='gray',vmin=0.9,vmax=1.1)
plt.axis('off')
np.savetxt('average_div_image.txt', average_image)
plt.savefig('average_div_image.png', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

