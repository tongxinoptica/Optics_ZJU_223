from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

Tensor = transforms.ToTensor()
Tensor2 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


class train_data(Dataset):

    def __init__(self, root_dir, label_dir):
        # super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        # self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.root_dir)
        self.truth_img_path = os.listdir(self.label_dir)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        truth_img_name = self.truth_img_path[idx]
        img_path = os.path.join(self.root_dir, img_name)
        truth_img_path = os.path.join(self.label_dir, truth_img_name)
        img = Image.open(img_path)
        img = Tensor2(img)
        # img = img/(torch.max(img))

        truth_img = Image.open(truth_img_path)
        truth_img = Tensor2(truth_img)


        return img, truth_img

    def __len__(self):
        return len(self.img_path)

"""
if __name__ == '__main__':
    root_dir = 'D:/train/input_data/all'
    label_dir = 'D:/train/groundtruth_data/all'
    train_dataset = train_data(root_dir, label_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    writer = SummaryWriter("test1")
    step = 0
    for batch_id, data in enumerate(train_dataloader):
        imgs, label = data  
        # print(batch_id)
        # print(imgs.size())
        # print(label.size())
        writer.add_images("imgs", imgs, step)  
        writer.add_images("label", label, step)
        step += 1
    writer.close()

    # roses_dataset = Tenser(roses_dataset)
    # roses_dataloader = DataLoader(roses_dataset, batch_size=4)
    # for data in enumerate(roses_dataloader):
    #     imgs, labels = data
    #
    #     print(labels)
"""
