import torch
import torchvision
from torchvision import transforms
from model import RDR_model
from tensorboardX import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


Tensor = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
RDR_model = RDR_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 定义训练的设备
RDR_model.to(device)
print("模型加载完成")

Image_path = 'D:/picture/total/beifen/2f/in_img.jpg'
img = Image.open(Image_path)
img = Tensor(img)
print(img.size())
img = torch.reshape(img, (1, 1, 256, 256))
RDR_model.load_state_dict(torch.load('./parameter/mse_gr36_2f'))
print(RDR_model.named_parameters())

# for name, para in RDR_model.named_parameters():
#     para.to(device)
#     if 'conv' in name and 'weight' in name and len(para.size()) == 4:
#         if name == 'RDDB4.conv2.weight':  #'RDDB2.conv2.weight':  #'RB1.conv1.weight':
#             k_w = para.size()[3]  # 卷积核宽
#             k_h = para.size()[2]
#             in_channel = para.size()[1]
#             out_channel = para.size()[0]
#             k_all = para.view(-1, 1, k_w, k_h)
#             k_all = torch.squeeze(k_all, 1)
#             k_all.to(device)
#             num = 256*256
#             print(name)
#             print(para.size())
#             print(k_all.size())
#             for c in range(1,num+1):
#                 #plt.subplot(2,4,c)
#                 #plt.subplots_adjust(wspace=0, hspace=0)
#                 to_pil = transforms.ToPILImage()
#                 output_pil = to_pil(k_all[c-1])
#                 plt.imshow(output_pil, cmap='gray')
#                 plt.axis('off')
#                 plt.savefig("./feature/2f/img/rddb4_f/{}".format(c), bbox_inches='tight', pad_inches=0)
#                 if c==2560:
#                     break
#             plt.show()

# model_child = list(RDR_model.children())
# print(type(model_child[1][1].children()))

b = torch.ones(256, 256).to(device)
RDR_model.eval()
with torch.no_grad():
    img = img.to(device)
    output = RDR_model(img)
    output = torch.where(output < 1, output, b)
    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output[0])  # 删去batch维度
    print(output.size())
    plt.imshow(output_pil, cmap='gray')
    plt.axis('off')
    plt.show()
