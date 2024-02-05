import torch
from torchvision import transforms
from model import RDR_model
from unit import to_psnr, to_ssim_skimage, to_ssim, to_pearson
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

Tensor = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
RDR_model = RDR_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 定义训练的设备
RDR_model.to(device)
print("模型加载完成")

# Image_path = 'D:/picture/robustness/translation/1.5f-80.jpg'
# label_path = 'D:/picture/robustness/gt.jpg'
Image_path = 'in_voc.jpg'
label_path = 'gt_voc.jpg'
img = Image.open(Image_path).convert('L')
img = Tensor(img)
img = torch.reshape(img, (1, 1, 256, 256))
label = Image.open(label_path)
label = Tensor(label)
label = torch.reshape(label, (1, 1, 256, 256))  # 增加batch维度
RDR_model.load_state_dict(torch.load('./parameter/mse_gr36_f'))
b = torch.ones(256, 256).to(device)

RDR_model.eval()
with torch.no_grad():
    img = img.to(device)
    label = label.to(device)
    output = RDR_model(img)
    output = torch.where(output < 1, output, b)
    # print(torch.max(output))
    mseloss = nn.MSELoss()
    mse = mseloss(output, label)
    psnr = to_psnr(output, label)
    npcc = to_pearson(output, label)
    print(psnr)
    print(mse)
    print(npcc)
    ssim = to_ssim_skimage(output, label)
    ssim1 = to_ssim(output, label)
    print(ssim)
    #print(ssim1)
    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output[0])  # 删去batch维度
    plt.imshow(output_pil, cmap='gray')
    plt.axis('off')
    #plt.savefig('D:/out1.5f-80.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()
