import argparse
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from unit import to_mseloss, to_ssim, to_psnr, to_pearson
from Diffraction_H import Diffraction_propagation, get_amplitude, get_phase, get_hologram, get_0_2pi, SLM_Propagation

def main(args):
    wavelength = args.wavelength
    dx = args.dx
    samples = args.samples
    dis = args.dis
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_path = 'D:/onn_image/f/0cm/10.jpg'
    image = Image.open(img_path)
    # Show padding image
    plt.imshow((image), cmap='gray')
    plt.axis('off')
    plt.title('Input')
    plt.show()

    image = to_tensor(image)
    input = image.unsqueeze(0).to(device)
    phase1 = np.loadtxt('./parameter/slm_0-5_phase.txt')
    p = torch.tensor(phase1).unsqueeze(0).unsqueeze(0).to(device)

    output = SLM_Propagation(input**0.5, dis, dx, wavelength, p)
    am = get_amplitude(output)
    intensity = am**2
    # intensity = intensity / torch.max(intensity)  # Normol
    print('output_mse: {}'.format(to_mseloss(intensity, input)))
    print('output_psnr: {}'.format(to_psnr(intensity, input)))
    print('output_ssim: {}'.format(to_ssim(intensity, input)))
    plt.imshow(to_pil(intensity[0]), cmap='gray')
    plt.title('Output')
    plt.axis('off')
    plt.show()

    # plt.imsave('Phase.jpg', to_pil(intensity[0]), cmap='gray')

    # No phase
    output_none = Diffraction_propagation(input, dis, dx, wavelength,
                                          transfer_fun='Angular Spectrum')
    am_none = get_amplitude(output_none)
    am_none = am_none**2
    # am_none = am_none / torch.max(am_none)
    print('none_mse: {}'.format(to_mseloss(am_none, input)))
    print('none_psnr: {}'.format(to_psnr(am_none, input)))
    print('nonet_ssim: {}'.format(to_ssim(am_none, input)))
    plt.imshow(to_pil(am_none[0]), cmap='gray')
    plt.title('None')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavelength', type=float, default=532e-9)  # Unit:m
    parser.add_argument('--dx', type=float, default=8e-6)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--dis', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--val-batch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load_parameter', type=bool, default=False)
    opt = parser.parse_args()
    main(opt)