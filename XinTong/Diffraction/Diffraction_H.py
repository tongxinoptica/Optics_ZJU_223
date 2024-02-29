import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from unit import to_mseloss, to_ssim, to_pearson, to_psnr
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Diffraction_propagation(field, distance, dx, wavelength, transfer_fun='Angular Spectrum'):
    H = get_transfer_fun(
        field.shape[-2],
        field.shape[-1],
        dx=dx,
        wavelength=wavelength,
        distance=distance,
        transfer_fun=transfer_fun,
        device=device)
    U1 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))
    U2 = U1 * H
    result = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U2)))
    return result


def get_transfer_fun(nu, nv, dx, wavelength, distance, transfer_fun, device=device):
    distance = torch.tensor([distance]).to(device)
    fx = torch.linspace(-1. / 2. / dx, 1. / 2. / dx - 1 / (2 * dx * nu), nu, dtype=torch.float32, device=device)
    fy = torch.linspace(-1. / 2. / dx, 1. / 2. / dx - 1 / (2 * dx * nv), nv, dtype=torch.float32, device=device)
    FY, FX = torch.meshgrid(fx, fy)
    if transfer_fun == 'Angular Spectrum':
        H = torch.exp(1j * distance * 2 * (
                np.pi * (1 / wavelength) * torch.sqrt(1. - (wavelength * FX) ** 2 - (wavelength * FY) ** 2)))
        H = H.to(device)
        return H
    if transfer_fun == 'Fresnel':
        k = 2 * np.pi * (1 / wavelength)
        H = torch.exp(1j * k * distance * (1 - 0.5 * ((FX * wavelength) ** 2 + (FY * wavelength) ** 2)))
        H = H.to(device)
        return H


def get_amplitude(field):
    Amplitude = torch.abs(field)
    return Amplitude


def get_phase(field):  # --> 0-2pi
    Phase = field.imag.atan2(field.real)
    Phase = (Phase + 2 * np.pi) % (2 * np.pi)
    return Phase


def get_hologram(amplitude, phase):
    hologram = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
    return hologram


def get_0_2pi(phase):
    return phase % (2 * np.pi)


def ONN_Propagation(img, dis_first, dis_onn, dis_after, dx, wavelength, p1, p2, p3, p4):
    output1 = Diffraction_propagation(img, dis_first, dx, wavelength, transfer_fun='Angular Spectrum')
    am1 = get_amplitude(output1)
    ph1 = get_phase(output1) + p1
    ph1 = get_0_2pi(ph1)
    hologram1 = get_hologram(am1, ph1)

    # Propagation 2
    output2 = Diffraction_propagation(hologram1, dis_onn, dx, wavelength, transfer_fun='Angular Spectrum')
    am2 = get_amplitude(output2)
    ph2 = get_phase(output2) + p2
    ph2 = get_0_2pi(ph2)
    hologram2 = get_hologram(am2, ph2)

    # Propagation 3
    output3 = Diffraction_propagation(hologram2, dis_onn, dx, wavelength, transfer_fun='Angular Spectrum')
    am3 = get_amplitude(output3)
    ph3 = get_phase(output3) + p3
    ph3 = get_0_2pi(ph3)
    hologram3 = get_hologram(am3, ph3)

    # Propagation 4
    output4 = Diffraction_propagation(hologram3, dis_onn, dx, wavelength, transfer_fun='Angular Spectrum')
    am4 = get_amplitude(output4)
    ph4 = get_phase(output2) + p4
    ph4 = get_0_2pi(ph4)
    hologram4 = get_hologram(am4, ph4)

    # Propagation last
    output5 = Diffraction_propagation(hologram3, dis_after, dx, wavelength, transfer_fun='Angular Spectrum')
    return output5

def SLM_Propagation(img, dis, dx, wavelength, p):
    img = get_hologram(img, p)
    output = Diffraction_propagation(img, dis, dx, wavelength)
    return output


'''
# 生成矩形孔的函数
def rect_function(x, a):
    return torch.where(torch.abs(x) <= a / 2, 1, 0)
# field = rect_function(X, Lx_size) * rect_function(Y, Ly_size)
Lx_size = 5e-3
Ly_size = 5e-3
'''

'''
wavelength = 532e-9  # m
dx = 2 * 8e-6
samples = 1000
L = dx * samples
dis_first = 0.2
dis_onn = 0.05
dis_after = 0.3
k = 2 * np.pi / wavelength
x = torch.linspace(-0.5 * L, 0.5 * L, samples)
y = torch.linspace(-0.5 * L, 0.5 * L, samples)
X, Y = torch.meshgrid(x, y)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Original image
img_path = 'D:/HR_data/process_sphe/celeb_process/3.bmp'
image = Image.open(img_path).convert('L').resize((720, 720))  # (1,720,720) PIL
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.show()

# Padding 0 to 1000*1000
image = to_tensor(image)
pad_left = (1000 - 720) // 2
pad_right = 1000 - 720 - pad_left
pad_top = (1000 - 720) // 2
pad_bottom = 1000 - 720 - pad_top
padding_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

# Show padding image
plt.imshow(to_pil(padding_image), cmap='gray')
plt.axis('off')
# plt.title('Original')
plt.show()

# Calculate diffraction result
Input = padding_image.unsqueeze(0).to(device)
Output = Diffraction_propagation(Input, dis_first, dx, wavelength,
                                 transfer_fun='Angular Spectrum')  # Fresnel Angular Spectrum
amplitude = torch.abs(Output)
phase = Output.imag.atan2(Output.real)
amplitude = amplitude / torch.max(amplitude)

# Show and save Output
plt.imshow(to_pil(amplitude[0]), cmap='gray')
plt.axis('off')
# plt.colorbar()
plt.savefig('distance_{}_dx_{}.jpg'.format(dis_first, dx), bbox_inches='tight', pad_inches=0)
# plt.title('dis = {}'.format(dis_first))
plt.show()

# phase = (phase + 2 * np.pi) % (2 * np.pi)
# Input2 = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
# Output2 = Diffraction_propagation(Input2, dis_first, dx, wavelength, transfer_fun='Angular Spectrum')
# amplitude2 = torch.abs(Output2)
# amplitude2 = amplitude2 / torch.max(amplitude2)
# plt.imshow(to_pil(amplitude2[0]), cmap='gray')
# plt.axis('off')
# plt.title('-pi,pi,dis = 0.1')
# plt.show()

# Calculate index
mse = to_mseloss(Input, amplitude)
psnr = to_psnr(Input, amplitude)
ssim = to_ssim(Input, amplitude)
pcc = to_pearson(Input, amplitude)
print('mse = {}'.format(mse))
print('psnr = {}'.format(psnr))
print('ssim = {}'.format(ssim))
print('pcc = {}'.format(pcc))
'''
