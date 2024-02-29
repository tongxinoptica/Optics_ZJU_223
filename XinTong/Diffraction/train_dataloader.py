from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
rotation_degrees = (-5,5)
translate_fraction = (0.05, 0.05)
Tensor2 = transforms.Compose([transforms.RandomRotation(rotation_degrees),
                              transforms.RandomAffine(degrees=0, translate=translate_fraction, fill=0),
                              transforms.ToTensor()])
Tensor = transforms.ToTensor()


class train_data(Dataset):  # 定义一个类，用Dataset去继承

    def __init__(self, root_dir, label_dir):  # 初始化类，定义一些变量
        # super().__init__()
        self.root_dir = root_dir  # 让其变成全局变量，可以在这一类中使用, 根路径
        self.label_dir = label_dir  # 路径下的分路径
        # self.path = os.path.join(self.root_dir, self.label_dir)  # 将其拼接成一个完整的路径
        self.img_path = os.listdir(self.root_dir)  # 读取该整个路径下的东西
        self.truth_img_path = os.listdir(self.label_dir)

    def __getitem__(self, idx):  # idx用作给予编号，列表形式
        img_name = self.img_path[idx]  # 在该整个路径下，把编号为idx的名称赋予给img_name
        truth_img_name = self.truth_img_path[idx]
        img_path = os.path.join(self.root_dir, img_name)  # 把路径下每一张图片都给予一个对应的路径
        truth_img_path = os.path.join(self.label_dir, truth_img_name)
        img = Image.open(img_path)  # 打开地址下的图片
        img = Tensor(img)
        # img = img/(torch.max(img))  # 归一化0-1
        truth_img = Image.open(truth_img_path)
        truth_img = Tensor(truth_img)
        return img, truth_img  # 不能返回全局变量，所以需要赋予给label

    def __len__(self):
        return len(self.img_path)  # 返回该路径下的东西的长度


# if __name__ == '__main__':
#     root_dir = 'D:/train/input_data/all'
#     label_dir = 'D:/train/groundtruth_data/all'
#     train_dataset = train_data(root_dir, label_dir)
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
#
#     writer = SummaryWriter("test1")
#     step = 0
#     for batch_id, data in enumerate(train_dataloader):
#         imgs, label = data  # 把dataloader打包好的数据送给img和label
#         # print(batch_id)
#         # print(imgs.size())
#         # print(label.size())
#         writer.add_images("imgs", imgs, step)  # 标题， 图片， 序号 有batch_size用add_images
#         writer.add_images("label", label, step)
#         step += 1
#     writer.close()
