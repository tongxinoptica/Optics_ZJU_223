import argparse
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from unit import to_mseloss, to_ssim, to_psnr, to_pearson
from Diffraction_H import Diffraction_propagation, get_amplitude, get_phase, get_hologram, get_0_2pi, ONN_Propagation


def main(args):
    wavelength = args.wavelength
    dx = args.dx
    samples = args.samples
    dis_first = args.dis_first
    dis_onn = args.dis_onn
    dis_after = args.dis_after

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Original image
    img_path = 'D:/HR_data/process_sphe/celeb_process/3.bmp'  # './F_mnist/test/30.jpg'
    image = Image.open(img_path).convert('L').resize((720, 720))  # (1,720,720) PIL

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
    plt.title('Original')
    plt.show()

    # Input image
    input = padding_image.unsqueeze(0).to(device)

    # Load phase
    phase1 = np.loadtxt('./parameter/p_3_10_10_10/p1.txt')
    phase2 = np.loadtxt('./parameter/p_3_10_10_10/p2.txt')
    phase3 = np.loadtxt('./parameter/p_3_10_10_10/p3.txt')
    phase4 = np.loadtxt('./parameter/p_4_5_15_10/p4.txt')
    p1 = torch.tensor(phase1).unsqueeze(0).unsqueeze(0).to(device)
    p2 = torch.tensor(phase2).unsqueeze(0).unsqueeze(0).to(device)
    p3 = torch.tensor(phase3).unsqueeze(0).unsqueeze(0).to(device)
    p4 = torch.tensor(phase4).unsqueeze(0).unsqueeze(0).to(device)

    # Propagation
    output = ONN_Propagation(input, dis_first, dis_onn, dis_after, dx, wavelength, p1, p2, p3, p4)
    am5 = get_amplitude(output)
    am5 = am5 / torch.max(am5)
    print('output_mse: {}'.format(to_mseloss(am5, input)))
    print('output_psnr: {}'.format(to_psnr(am5, input)))
    print('output_ssim: {}'.format(to_ssim(am5, input)))
    plt.imshow(to_pil(am5[0]), cmap='gray')
    plt.title('Phase')
    plt.axis('off')
    plt.show()
    plt.imsave('Phase.jpg', to_pil(am5[0]), cmap='gray')

    # No phase
    output_none = Diffraction_propagation(input, dis_first + dis_after + 3 * dis_onn, dx, wavelength,
                                          transfer_fun='Angular Spectrum')
    am_none = get_amplitude(output_none)
    am_none = am_none / torch.max(am_none)
    print('none_mse: {}'.format(to_mseloss(am_none, input)))
    print('none_psnr: {}'.format(to_psnr(am_none, input)))
    print('nonet_ssim: {}'.format(to_ssim(am_none, input)))
    plt.imshow(to_pil(am_none[0]), cmap='gray')
    plt.title('None')
    plt.axis('off')
    plt.show()
    plt.imsave('NoPhase.jpg', to_pil(am_none[0]), cmap='gray')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavelength', type=float, default=532e-9)  # Unit:m
    parser.add_argument('--dx', type=float, default=8e-6)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--dis_first', type=float, default=0.1)
    parser.add_argument('--dis_onn', type=float, default=0.05)
    parser.add_argument('--dis_after', type=float, default=0.1)
    opt = parser.parse_args()
    main(opt)
