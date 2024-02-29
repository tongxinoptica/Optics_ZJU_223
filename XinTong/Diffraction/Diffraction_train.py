import numpy as np
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
import sys
import torch.nn as nn
from Diffraction_H import get_amplitude, ONN_Propagation
from unit import to_ssim, to_pearson, to_psnr


def main(args):
    from torch.utils.data import DataLoader
    from val_dataloader import val_data
    from train_dataloader import train_data
    wavelength = args.wavelength
    dx = args.dx
    samples = args.samples
    dis_first = args.dis_first
    dis_onn = args.dis_onn
    dis_after = args.dis_after
    train_batch = args.batch_size
    val_batch = args.val_batch
    learning_rate = args.lr
    total_epoch = args.epochs
    Load_parameter = args.load_parameter
    L = dx * samples
    # x = torch.linspace(-0.5 * L, 0.5 * L, samples)
    # y = torch.linspace(-0.5 * L, 0.5 * L, samples)
    # X, Y = torch.meshgrid(x, y)

    # Load training data
    train_root_dir = 'D:/HR_data/exp_data/train_gt'
    train_label_dir = 'D:/HR_data/exp_data/train_gt'
    train_loader = DataLoader(train_data(train_root_dir, train_label_dir), batch_size=train_batch, shuffle=True)

    # Load celeb test data
    celeb_val_root_dir = 'D:/HR_data/exp_data/test_gt/celeb'
    celeb_val_label_dir = 'D:/HR_data/exp_data/test_gt/celeb'
    celeb_val_loader = DataLoader(val_data(celeb_val_root_dir, celeb_val_label_dir), batch_size=val_batch)

    # Define training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loss function
    loss_function = nn.MSELoss()

    # Generate parameters --> (0, 2pai)
    if Load_parameter:
        print('Parameter loaded')
        phase1 = np.loadtxt('./parameter/p1.txt')
        phase2 = np.loadtxt('./parameter/p2.txt')
        phase3 = np.loadtxt('./parameter/p3.txt')
        phase4 = np.loadtxt('./parameter/p4.txt')
        p1 = torch.tensor(phase1).unsqueeze(0).unsqueeze(0).to(device)
        p2 = torch.tensor(phase2).unsqueeze(0).unsqueeze(0).to(device)
        p3 = torch.tensor(phase3).unsqueeze(0).unsqueeze(0).to(device)
        p4 = torch.tensor(phase4).unsqueeze(0).unsqueeze(0).to(device)
        p1 = torch.nn.Parameter(p1)
        p2 = torch.nn.Parameter(p2)
        p3 = torch.nn.Parameter(p3)
        p4 = torch.nn.Parameter(p4)
    else:
        param_size = (1, 1, 1000, 1000)
        p1 = torch.nn.Parameter(torch.zeros(param_size, dtype=torch.float, requires_grad=True).to(device))
        p2 = torch.nn.Parameter(torch.zeros(param_size, dtype=torch.float, requires_grad=True).to(device))
        p3 = torch.nn.Parameter(torch.zeros(param_size, dtype=torch.float, requires_grad=True).to(device))
        p4 = torch.nn.Parameter(torch.zeros(param_size, dtype=torch.float, requires_grad=True).to(device))

    # Optimizer
    Optimizer = optim.Adam([p1, p2, p3, p4], learning_rate)

    # initial setting
    old_mse = 0.1
    # psnr_list = []
    # ssim_list = []
    mse_list = []
    # npcc_list = []
    celeb_mse_list = []

    for i in range(total_epoch):
        if (i + 1) % 25 == 0 and learning_rate >= 3e-8:
            learning_rate /= 2

        train_loader = tqdm(train_loader, file=sys.stdout)
        for batch_id, train_data in enumerate(train_loader):
            imgs, label = train_data
            imgs = imgs.to(device)
            label = label.to(device)
            output = ONN_Propagation(imgs, dis_first, dis_onn, dis_after, dx, wavelength, p1, p2, p3, p4)
            am5 = get_amplitude(output)
            loss = loss_function(am5, label)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

            # Limit weight range
            p1.data = torch.clamp(p1.data, 0., 2 * np.pi)
            p2.data = torch.clamp(p2.data, 0., 2 * np.pi)
            p3.data = torch.clamp(p3.data, 0., 2 * np.pi)
            p4.data = torch.clamp(p4.data, 0., 2 * np.pi)

            mse_list.append(loss.item())
            # psnr_list.append(to_psnr(am5, label))
            # ssim_list.append(to_ssim(am5, label))
            # npcc_list.append(to_pearson(am5, label))
            train_loader.desc = "[train epoch {}] loss: {:.4f} lr:{}".format(i + 1, loss.item(), learning_rate)

        train_mse = sum(mse_list) / len(mse_list)
        # train_psnr = sum(psnr_list) / len(psnr_list)
        # train_ssim = sum(ssim_list) / len(ssim_list)
        # train_npcc = sum(npcc_list) / len(npcc_list)
        print("Average train mse:{}".format(train_mse))
        # print("Average train psnr:{}".format(train_psnr))
        # print("Average train ssim:{}".format(train_ssim))
        # print("Average train npcc:{}".format(train_npcc))

        for batch_id, val_data in enumerate(celeb_val_loader):
            with torch.no_grad():
                imgs, label = val_data
                imgs = imgs.to(device)
                label = label.to(device)
                output = ONN_Propagation(imgs, dis_first, dis_onn, dis_after, dx, wavelength, p1, p2, p3, p4)
                am = get_amplitude(output)

            celeb_mse_list.append(loss_function(am, label).item())

        celeb_test_mse = sum(celeb_mse_list) / len(celeb_mse_list)
        print("Average celeb test mse:{}".format(celeb_test_mse))
        if celeb_test_mse < old_mse:
            old_mse = celeb_test_mse
            best_phase1 = p1.data.clone()
            best_phase2 = p2.data.clone()
            best_phase3 = p3.data.clone()
            best_phase4 = p4.data.clone()

            p1_np = best_phase1.cpu().detach().numpy().reshape(1000, 1000)
            p2_np = best_phase2.cpu().detach().numpy().reshape(1000, 1000)
            p3_np = best_phase3.cpu().detach().numpy().reshape(1000, 1000)
            p4_np = best_phase4.cpu().detach().numpy().reshape(1000, 1000)

            np.savetxt('./parameter/densep4/p1.txt', p1_np)
            np.savetxt('./parameter/densep4/p2.txt', p2_np)
            np.savetxt('./parameter/densep4/p3.txt', p3_np)
            np.savetxt('./parameter/densep4/p4.txt', p4_np)

            print('Weight Update')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavelength', type=float, default=532e-9)  # Unit:m
    parser.add_argument('--dx', type=float, default=8e-6)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--dis_first', type=float, default=0.1)
    parser.add_argument('--dis_onn', type=float, default=0.05)
    parser.add_argument('--dis_after', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--load_parameter', type=bool, default=False)
    opt = parser.parse_args()
    main(opt)
