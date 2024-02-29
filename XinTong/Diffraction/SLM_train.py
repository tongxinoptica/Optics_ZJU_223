import numpy as np
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
import sys
import torch.nn as nn
from Diffraction_H import get_amplitude, SLM_Propagation


def main(args):
    from torch.utils.data import DataLoader
    from val_dataloader import val_data
    from train_dataloader import train_data
    wavelength = args.wavelength
    dx = args.dx
    samples = args.samples
    dis = args.dis
    train_batch = args.batch_size
    val_batch = args.val_batch
    learning_rate = args.lr
    total_epoch = args.epochs
    Load_parameter = args.load_parameter
    L = dx * samples

    # Load training data
    train_root_dir = 'D:/onn_image/f/0cm'
    train_label_dir = 'D:/onn_image/gt_train'
    train_loader = DataLoader(train_data(train_root_dir, train_label_dir), batch_size=train_batch, shuffle=True)

    # Load celeb test data
    celeb_val_root_dir = 'D:/onn_image/f/0cm_val'
    celeb_val_label_dir = 'D:/onn_image/gt_val'
    celeb_val_loader = DataLoader(val_data(celeb_val_root_dir, celeb_val_label_dir), batch_size=val_batch)

    # Define training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loss function
    loss_function = nn.MSELoss()

    param_size = (1, 1, 1000, 1000)
    p = torch.nn.Parameter(torch.zeros(param_size, dtype=torch.float, requires_grad=True).to(device))
    Optimizer = optim.Adam([p], learning_rate)

    old_mse = 0.3
    train_mse_list = []
    val_mse_list = []

    for i in range(total_epoch):
        if (i + 1) % 25 == 0 and learning_rate >= 3e-8:
            learning_rate /= 2

        train_loader = tqdm(train_loader, file=sys.stdout)
        for batch_id, train_data in enumerate(train_loader):
            imgs, label = train_data
            imgs = (imgs.to(device)) ** 0.5  # Amplitude
            label = label.to(device)
            output = SLM_Propagation(imgs, dis, dx, wavelength, p)
            am = get_amplitude(output)
            intensity = am ** 2
            # intensity = intensity / torch.max(intensity)  # Normol
            loss = loss_function(intensity, label)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            # p.data = torch.clamp(p.data, 0., 2 * np.pi)
            train_mse_list.append(loss.item())
            train_loader.desc = "[train epoch {}] loss: {:.4f} lr:{}".format(i + 1, loss.item(), learning_rate)
        train_mse = sum(train_mse_list) / len(train_mse_list)
        print("Average train mse:{}".format(train_mse))

        for batch_id, val_data in enumerate(celeb_val_loader):
            with torch.no_grad():
                imgs, label = val_data
                imgs = (imgs.to(device)) ** 0.5
                label = label.to(device)
                output = SLM_Propagation(imgs, dis, dx, wavelength, p)
                am = get_amplitude(output)
                intensity = am ** 2
            val_mse_list.append(loss_function(intensity, label).item())

        val_mse = sum(val_mse_list) / len(val_mse_list)
        print("Average val mse:{}".format(val_mse))

        if val_mse < old_mse:
            old_mse = val_mse
            best_phase = p.data.clone()
            slm_phase = best_phase.cpu().detach().squeeze().numpy()
            np.savetxt('./parameter/slm_0-5_phase.txt', slm_phase)
            print('Weight Update')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavelength', type=float, default=532e-9)  # Unit:m
    parser.add_argument('--dx', type=float, default=8e-6)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--dis', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load_parameter', type=bool, default=False)
    opt = parser.parse_args()
    main(opt)
