import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Dense block
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        # self.bn2 = nn.BatchNorm2d(growth_rate)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=1)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        # out = self.bn2(out)
        # out = self.relu2(out)
        # out = self.conv2(out)
        out = torch.cat((x, out), 1)
        return out


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(conv_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        # if out.size(1)==16:
        #     for c in range(1,17):
        #         plt.subplot(4,4,c)
        #         plt.subplots_adjust(wspace=0, hspace=0)
        #         feature = out.squeeze(0)
        #         to_pil = transforms.ToPILImage()
        #         output_pil = to_pil(feature[c-1])
        #         plt.imshow(output_pil, cmap='gray')
        #         plt.axis('off')
        #         #plt.savefig("./feature/2f/voc/f_cb{}.jpg".format(c), bbox_inches='tight', pad_inches=0)
        #     plt.show()
        return out


#  Residual block
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Residual_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        identity = x
        identity = self.conv1(self.bn1(identity))
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        # if out.size(1)==8:
        #     for c in range(1,9):
        #         plt.subplot(2,4,c)
        #         plt.subplots_adjust(wspace=0,hspace=0)
        #         feature = out.squeeze(0)
        #         to_pil = transforms.ToPILImage()
        #         output_pil = to_pil(feature[c-1])
        #         plt.imshow(output_pil, cmap='gray')
        #         plt.axis('off')
        #         #plt.savefig("./feature/2f/voc/f_rb{}.jpg".format(c), bbox_inches='tight', pad_inches=0)
        #     plt.show()
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = out + identity

        return out


# model = Residual_Block(1, 8)
# print(model)


# Residual Dense Block
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.bn2 = nn.BatchNorm2d(_in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv_1x1(out)
        out = out + x
        return out


#  RDDB Layer
class RD_Down_layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_dense_layer, growth_rate, kernel_size=3):
        super(RD_Down_layer, self).__init__()
        self.RDB = RDB(in_channels, num_dense_layer=num_dense_layer, growth_rate=growth_rate)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # self.drop = nn.Dropout(0.05)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = self.RDB(x)
        # identity = out
        # identity = self.conv1(self.bn1(identity))

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # out = self.drop(out)
        out = self.conv2(out)
        # out = out + identity
        return out


#  RDUB Layer
class RD_Up_layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_dense_layer, growth_rate, kernel_size=3):
        super(RD_Up_layer, self).__init__()
        self.RDB = RDB(in_channels, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.convT1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # self.drop = nn.Dropout(0.05)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        out = self.RDB(x)
        # identity = out
        # identity = self.convT1(self.bn1(identity))
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.convT1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # out = self.drop(out)
        out = self.conv2(out)
        # out = out + identity
        return out


class RD_Residual_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_dense_layer=3, growth_rate=36):
        super(RD_Residual_Layer, self).__init__()
        self.RDB = RDB(in_channels, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.residual = Residual_Block(in_channels, out_channels)

    def forward(self, x):
        out = self.RDB(x)
        out = self.residual(out)
        return out

# model1 = RD_Residual_Layer(in_channels=24, out_channels=8)
# model2 = RD_Down_layer(in_channels=8)
# model3 = RD_Up_layer(32, 16)
# model4 = Residual_Block(8, 1)
# print(model1)
# print(model2)
# print(model3)
# print(model4)

# img_path = "C:/Users/Administrator/Desktop/practice/1.jpg"
# img = Image.open(img_path)
# transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
# img = transform(img)
# img = torch.reshape(img, (1, 3, 256, 256))
#
# model = RD_Up_layer(3)
# img = model(img)
# print(img.shape)
