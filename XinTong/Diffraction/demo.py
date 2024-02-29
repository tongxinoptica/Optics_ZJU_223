import os
import shutil

from PIL import Image


input_path = 'D:/onn_image/f/0cm'
new_path = 'D:/onn_image/f/0cm_val'
if not os.path.exists(new_path):
    os.makedirs(new_path)
img_names = os.listdir(input_path)
img_names.sort(key=lambda x: int(x[:-4]))
for j in range(10,4750,10):
    img_path = os.path.join(input_path,img_names[j])
    shutil.move(img_path, new_path)


