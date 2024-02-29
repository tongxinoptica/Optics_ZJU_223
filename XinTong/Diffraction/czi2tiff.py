from aicsimageio import AICSImage
import numpy as np
import os
import tifffile
from pathlib import Path
from PIL import Image
from skimage import feature, morphology, img_as_float
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt


def convert_czi_to_tiff_mip(input_folder, output_folder):
    # 确保输出文件夹存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.czi'):
            # 创建完整的文件路径
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name.replace('.czi', '-MIP.tiff'))

            # 读取 czi 文件
            img = AICSImage(input_file)
            img_virtual = img.get_image_dask_data("ZYX", T=0, C=0)  # select channel
            img_real = img_virtual.compute()  # read in-memory
            img_MIP = np.max(img_real, 0)  # max intensity projection in z dimension

            # 保存为 tiff 文件
            tifffile.imwrite(output_file, img_MIP)

            print(f'Converted {input_file} to {output_file}')


def process_image(input_image_path, output_binary_path, output_edge_path, line_width):
    # 打开图像文件并转换为灰度图像
    for file_name in os.listdir(input_image_path):
        input_image = os.path.join(input_image_path, file_name)
        output_binary = os.path.join(output_binary_path, f"binary_{file_name}")
        output_edge = os.path.join(output_edge_path, f"edge_{file_name}")
        img = Image.open(input_image)
        img_array = np.array(img)
        bit_depth = img.mode
        if bit_depth == 'I;16':  # 16bit 图像
            img_array = ((img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min()))).astype('uint8')


        img_array[img_array > 128] = 255
        img_binary = Image.fromarray(img_array)
        img_binary.save(output_binary)
        # 将图像数据类型转换为float，因为feature.canny期望float类型的输入
        img_float = img_as_float(img_array)
        # 使用Canny算法检测边缘
        edges = feature.canny(img_float, sigma=3.5)
        if line_width > 1:
            edges = morphology.dilation(edges, morphology.disk(line_width // 2))

        # 创建一个红色边缘图层
        edges_color = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        edges_color[edges] = [255, 0, 0]  # 将边缘设置为红色

        # 将红色边缘图层叠加在原始图像上
        img_color = np.dstack((img_array, img_array, img_array))  # 将灰度图转换为RGB
        img_with_edges = np.where(edges_color, edges_color, img_color)  # 将红色边缘叠加到图像上
        img_with_edges = Image.fromarray(img_with_edges)
        img_with_edges.save(output_edge)
        print('finish')

    # 使用二值形态学操作填充边缘内的区域
    # filled_cells = binary_fill_holes(edges)
    # 将原图像中对应mask内的区域的像素值设置为0
    # img_array[filled_cells] = 0
    # 再次将numpy数组转换为PIL图像
    # img_processed = Image.fromarray(img_array)
    # 显示处理后的图像
    # img_processed.show()

    # 保存处理后的图像为jpg格式
    # img_processed.save(output_image_path, 'JPEG')


# 设置输入输出路径
input_image_path = 'C:/Users/Administrator/Desktop/20240118/20240118-4tiff'
output_binary_path = 'C:/Users/Administrator/Desktop/20240118/20240118-4binary'
output_edge_path = 'C:/Users/Administrator/Desktop/20240118/20240118-4edge'
process_image(input_image_path, output_binary_path, output_edge_path, line_width=3)

# 设置输入输出路径
# input_folder = 'C:/Users/Administrator/Desktop/20240118/20240118-4'
# output_folder = 'C:/Users/Administrator/Desktop/20240118/20240118-4tiff'
