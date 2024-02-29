import struct
import numpy as np
import cv2
import os

import torch

'''
img_file =  r't10k-images-idx3-ubyte'  #  Path to your xxx-images-idx3-ubyte
label_file = r't10k-labels-idx1-ubyte'  # Path to your xxx-labels-idx1-ubyte
generate_img_root = r".\F_mnist"    # Path to save the images


def check_file(img_path,label_path):
    _,key_img_name = os.path.split(img_path)
    key1 = key_img_name.split('-')

    _,key_label_name = os.path.split(label_path)
    key2 = key_label_name.split('-')
    if (key1[1] != key2[1] or key1[2] != key2[2]):
        raise ValueError("Please check your input file,"
                         "and make true they belong to the same class !!!")


def get_key_name(file_path):
    _,file_name = os.path.split(file_path)
    key_name = file_name.split('-')
    subdir_name = str(key_name[1]) + "_" + str(key_name[2])
    return subdir_name


def make_dir(file_root):
    if  not os.path.isdir(file_root):
        os.makedirs(file_root)

check_file(img_file,label_file)

generate_img_root = os.path.join(generate_img_root,get_key_name(img_file))
make_dir(generate_img_root)


binfile = open(label_file,'rb')
buf = binfile.read()
index = 0
_,train_label_num = struct.unpack_from('>II',buf,index)
index += struct.calcsize('>II')

train_label_lis=[]

for i in range(train_label_num):
    label_item = int(struct.unpack_from('>B',buf,index)[0])
    train_label_lis.append(label_item)
    index += struct.calcsize('B')

print(len(train_label_lis))



binfile_img = open(img_file,'rb')
buf_img = binfile_img.read()
image_index=0

_,train_img_num = struct.unpack_from('>II',buf_img,image_index)
print("train_img_num: ",train_img_num)
image_index += struct.calcsize('>IIII')
im_list = []

for i in range(train_img_num):
    im = struct.unpack_from('>784B' ,buf_img, image_index)
    im_list.append(np.reshape(im,(28,28)))
    image_index += struct.calcsize('>784B')


for i in range(len(train_label_lis)):
    label_name = str(train_label_lis[i])
    subdir = os.path.join(generate_img_root,label_name)
    make_dir(subdir)
    image_name = label_name+"_"+str(i)+'.png'
    img_path = os.path.join(subdir,image_name)
    cv2.imwrite(img_path, im_list[i])

print('generate done!')
print("Now rotating......")

list = os.listdir(generate_img_root)
for path in list:
    each_class_path = os.path.join(generate_img_root,path)
    img_list = os.listdir(each_class_path)
    for img in img_list:
        image_path = os.path.join(each_class_path,img)
        img = cv2.imread(image_path, 0)
        row, cols = img.shape
        M = cv2.getRotationMatrix2D((cols // 2, row // 2), -90, 1)
        res2 = cv2.warpAffine(img, M, (row, cols))
        res2 = cv2.flip(res2, 1)
        cv2.imwrite(image_path, res2)

print("Done")
'''

'''
import os
from skimage import io
import torchvision.datasets.mnist as mnist

root = ".\F_mnist"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)
# print("training set :", train_set[0].size())
print("test set :", test_set[0].size())


def convert_to_img(train=True):
    if train:
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):   #判断文件是否存在
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()


#convert_to_img(True)  # 转换训练集
convert_to_img(False)  # 转换测试集
'''



# import torch
# import torch.optim as optim
#
# # 定义两个可训练的参数
# p1 = torch.nn.Parameter(torch.tensor(1.0))  # 初始化为1.0
# p2 = torch.nn.Parameter(torch.tensor(1.0))  # 初始化为1.0
#
# # 将参数放入优化器
# optimizer = optim.Adam([p1, p2], lr=0.005)
#
#
# # 定义损失函数L
# def loss_func(p1, p2):
#     return (p1 ** 2 + p2 ** 2) + 3 * p1 * p2
#
#
# # 训练/优化循环
# num_epochs = 2000
# for epoch in range(num_epochs):
#     loss = loss_func(p1, p2)
#     optimizer.zero_grad()
#
#     loss.backward()
#     optimizer.step()
#     p1.data = torch.clamp(p1.data, 0.0, 1.0)
#     p2.data = torch.clamp(p2.data, 0.0, 1.0)
#
#     print(f"Optimized values: p1={p1.item()}, p2={p2.item()}")
#     print(loss)
from PIL import Image


from PIL import Image

def normalize_image(image):
    """Normalize the pixel values of an image to the range [0, 1]."""
    image = np.asarray(image).astype('float')
    return image / np.max(image)

def mse(imageA, imageB):
    """Calculate the mean squared error between two images."""
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    """Calculate the PSNR between two images."""
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # Max pixel value after normalization
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_value))

# Load and normalize two images
# Replace 'path_to_image_1.jpg' and 'path_to_image_2.jpg' with actual file paths
# imageA = Image.open("D:/turbulence/tur_image/ocean/100/339飞机/10.jpg")
# imageB = Image.open("D:/turbulence/tur_image/339.jpg")
# imageA = normalize_image(imageA.resize((256,256)))
# imageB = normalize_image(imageB.resize((256,256)))
# mse_value = mse(imageA, imageB)
# psnr_value = psnr(imageA, imageB)
# print(f"MSE: {mse_value}")
# print(f"PSNR: {psnr_value}")

def batch_rename_and_convert_to_jpg(folder_path, output_folder):
    """Rename all images in a folder sequentially starting from 1 and convert them to JPG."""
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the image counter
    img_counter = 1

    # Iterate over all image files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            old_path = os.path.join(folder_path, filename)
            new_filename = f"{img_counter}.jpg"  # New filename with JPG extension
            new_path = os.path.join(output_folder, new_filename)

            try:
                # Open and convert the image to JPG
                with Image.open(old_path) as img:
                    img = img.resize((900,900), Image.LANCZOS)
                    img.convert('L').save(new_path, 'JPEG', dpi=(100,100))

                img_counter += 1
            except Exception as e:
                print(f"Error processing {old_path}: {e}")

def batch_convert_to_binary(folder_path, output_folder):
    """Convert all grayscale images in a folder to binary images based on the average pixel value."""
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all image files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(output_folder, filename)

            try:
                with Image.open(old_path) as img:
                    # Ensure the image is in grayscale
                    # img_gray = img.convert('L')

                    # Convert the image to a NumPy array for computation
                    img_array = np.array(img)

                    # Calculate the average pixel value
                    threshold = img_array.mean()

                    # Apply binary thresholding
                    img_binary = (img_array > threshold) * 255

                    # Convert the NumPy array back to an image
                    img_binary = Image.fromarray(img_binary.astype('uint8'))

                    # Save the binary image
                    img_binary.save(new_path, 'JPEG')
            except Exception as e:
                print(f"Error processing {old_path}: {e}")



# batch_rename_and_convert_to_jpg('D:/HR_data/HR','D:/HR_data/HRgray')

batch_convert_to_binary('D:/HR_data/HR_gray', 'D:/HR_data/HR_Binary')









