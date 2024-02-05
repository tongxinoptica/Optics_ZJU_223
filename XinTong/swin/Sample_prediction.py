import torch
from torchvision import transforms
from swin_IR import TWC_Swin
# from unet_model import Unet
# from model import RDR_model
from unit import to_psnr, to_ssim_skimage, to_ssim, to_pearson
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

# Parameter setting
coherence = '1.2f'  # Change the coherence here

# Basic setting
Image_path = './Sample_images/sample_{}.jpg'.format(coherence)
label_path = './Sample_images/label.jpg'
Tensor = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
model = TWC_Swin()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
parameter_path = './parameter/lr=0.0005_{}_nor'.format(coherence)
print("Model loaded")

# Import image
img = Image.open(Image_path)
if img.mode == 'RGB':
    img = img.convert('L')
plt.imshow(img, cmap='gray')
plt.title('Original image', fontsize=20)
plt.axis('off')
plt.show()

# Preprocess
img = Tensor(img)
img = img / (torch.max(img))
img = torch.reshape(img, (1, 1, 128, 128))

label = Image.open(label_path)
if label.mode == 'RGB':
    label = label.convert('L')
label = Tensor(label)
label = label / (torch.max(label))
label = torch.reshape(label, (1, 1, 128, 128))  # 增加batch维度

model.load_state_dict(torch.load(parameter_path))
b = torch.ones(128, 128).to(device)

model.eval()
with torch.no_grad():
    img = img.to(device)
    label = label.to(device)
    output = model(img)
    output = torch.where(output < 1, output, b)
    mseloss = nn.MSELoss()
    mse = mseloss(output, label)
    psnr = to_psnr(output, label)
    npcc = to_pearson(output, label)
    print('MSE = {}'.format(mse))
    print('PSNR = {}'.format(psnr))
    ssim = to_ssim_skimage(output, label)
    print('SSIM = {}'.format(ssim[0]))
    print('NPCC = {}'.format(npcc))
    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output[0])  # 删去batch维度
    gt = to_pil(label[0])
    plt.imshow(output_pil, cmap='gray')
    plt.axis('off')
    plt.title('Output image', fontsize=20)
    plt.show()
