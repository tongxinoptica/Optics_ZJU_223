import torch
import argparse
import sys
import torch.nn as nn
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from unit import to_psnr, to_ssim_skimage, to_ssim, to_mseloss, to_pearson
from NPCC import NPCC_loss, ssim_loss, mse_loss
from tqdm import tqdm


def main(args):
    from train_dataloader import train_data
    from val_dataloader import val_data
    from swin_IR import swinIR

    train_batch = args.batch_size
    val_batch = args.val_batch
    learning_rate = args.lr
    total_epoch = args.epochs

    # Load training data
    train_root_dir = 'D:/HR_data/exp_data/{}/train'.format(args.coherence)
    train_label_dir = 'D:/HR_data/exp_data/train_gt'
    train_loader = DataLoader(train_data(train_root_dir, train_label_dir), batch_size=train_batch, shuffle=True)
    # Load BSD test data
    BSD_val_root_dir = 'D:/HR_data/exp_data/{}/test/BSD'.format(args.coherence)
    BSD_val_label_dir = 'D:/HR_data/exp_data/test_gt/BSD'
    BSD_val_loader = DataLoader(val_data(BSD_val_root_dir, BSD_val_label_dir), batch_size=val_batch)
    # Load celeb test data
    celeb_val_root_dir = 'D:/HR_data/exp_data/{}/test/celeb'.format(args.coherence)
    celeb_val_label_dir = 'D:/HR_data/exp_data/test_gt/celeb'
    celeb_val_loader = DataLoader(val_data(celeb_val_root_dir, celeb_val_label_dir), batch_size=val_batch)
    # Load DIV test data
    DIV_val_root_dir = 'D:/HR_data/exp_data/{}/test/DIV'.format(args.coherence)
    DIV_val_label_dir = 'D:/HR_data/exp_data/test_gt/DIV'
    DIV_val_loader = DataLoader(val_data(DIV_val_root_dir, DIV_val_label_dir), batch_size=val_batch)
    # Load flickr test data
    flickr_val_root_dir = 'D:/HR_data/exp_data/{}/test/flickr'.format(args.coherence)
    flickr_val_label_dir = 'D:/HR_data/exp_data/test_gt/flickr'
    flickr_val_loader = DataLoader(val_data(flickr_val_root_dir, flickr_val_label_dir), batch_size=val_batch)
    # Load WED test data
    WED_val_root_dir = 'D:/HR_data/exp_data/{}/test/WED'.format(args.coherence)
    WED_val_label_dir = 'D:/HR_data/exp_data/test_gt/WED'
    WED_val_loader = DataLoader(val_data(WED_val_root_dir, WED_val_label_dir), batch_size=val_batch)

    # Load model
    # swin_model = Unet()
    swin_model = swinIR()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 定义训练的设备
    swin_model.to(device)
    print(device)
    print("模型加载完成，开始训练")

    # 损失函数
    loss_function = nn.MSELoss()
    # loss_function3 = NPCC_loss()
    # loss_function1 = mse_loss()
    # loss_function2 = nn.SmoothL1Loss()
    # loss_function.to(device)

    # weight_decay_list = (param for name, param in RDR_model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    # no_decay_list = (param for name, param in RDR_model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    # parameters = [{'params': weight_decay_list},
    #               {'params': no_decay_list, 'weight_decay': 0.}]

    # optimizer = torch.optim.SGD(swin_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(swin_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(swin_model.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    # stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 加载权重
    # try:
    #     swin_model.load_state_dict(torch.load('parameter/lr=0.0005_1.7f_nor'))
    #     print('--- weight loaded ---')
    # except:
    #     print('--- no weight loaded ---')

    # old_psnr = 0.0
    # old_ssim = 0.0
    old_mse = 0.1
    # accumulation_steps = 4
    # b = torch.ones(256, 256).to(device)

    s = 1
    for i in range(total_epoch):
        if (i + 1) % 8 == 0 and learning_rate >= 3e-8:
            learning_rate /= 2

        # 后期加入学习率调整
        # 训练
        swin_model.train()
        psnr_list = []
        ssim_list = []
        mse_list = []
        npcc_list = []
        train_loader = tqdm(train_loader, file=sys.stdout)
        for batch_id, train_data in enumerate(train_loader):
            imgs, label = train_data
            imgs = imgs.to(device)
            label = label.to(device)

            output = swin_model(imgs)  # 计算输出
            # output = torch.where(output < 1, output, b)
            loss = loss_function(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss /= accumulation_steps
            # loss.backward()
            # if ((batch_id + 1) % accumulation_steps) == 0:  # 梯度叠加
            #     optimizer.step()
            #     optimizer.zero_grad()

            # Calculate train average psnr, ssim
            mse_list.append(loss.item())            # psnr_list.append(to_psnr(output, label))
            # ssim_list.append(to_ssim(output, label))
            # npcc_list.append(to_pearson(output, label))

            train_loader.desc = "[train epoch {}] loss: {:.4f} lr:{}".format(i + 1, loss.item(), learning_rate)
            # del imgs, label, output, loss
            # empty_cache()

        train_mse = sum(mse_list) / len(mse_list)
        # train_psnr = sum(psnr_list) / len(psnr_list)
        # train_ssim = sum(ssim_list) / len(ssim_list)
        # train_npcc = sum(npcc_list) / len(npcc_list)
        print("Average train mse:{}".format(train_mse))
        # print("Average train psnr:{}".format(train_psnr))
        # print("Average train ssim:{}".format(train_ssim))
        # print("Average train npcc:{}".format(train_npcc))
        s += 1
        # ws.cell(row=1, column=1).value = "train_mse"
        # ws.cell(row=1, column=2).value = "train_psnr"
        # ws.cell(row=1, column=3).value = "train_npcc"
        # ws.cell(row=1, column=4).value = "train_ssim"
        # ws.cell(row=s, column=1).value = train_mse.item()
        # ws.cell(row=s, column=2).value = train_psnr
        # ws.cell(row=s, column=3).value = train_npcc.item()
        # ws.cell(row=s, column=4).value = train_ssim.item()

        # 验证BSD数据集
        BSD_psnr_list_ = []
        BSD_ssim_list_ = []
        BSD_mse_list_ = []
        BSD_npcc_list_ = []

        swin_model.eval()
        for batch_id, val_data in enumerate(BSD_val_loader):
            with torch.no_grad():
                swin_model.eval()
                imgs, label = val_data
                imgs = imgs.to(device)
                label = label.to(device)
                output = swin_model(imgs)
                # output = torch.where(output < 1, output, b)

            # Calculate  eval average psnr, ssim

            BSD_psnr_list_.append(to_psnr(output, label))
            BSD_ssim_list_.append(to_ssim(output, label))
            BSD_mse_list_.append(to_mseloss(output, label))
            BSD_npcc_list_.append(to_pearson(output, label))

        BSD_val_psnr = sum(BSD_psnr_list_) / len(BSD_psnr_list_)
        BSD_val_ssim = sum(BSD_ssim_list_) / len(BSD_ssim_list_)
        BSD_val_mse = sum(BSD_mse_list_) / len(BSD_mse_list_)
        BSD_val_npcc = sum(BSD_npcc_list_) / len(BSD_npcc_list_)
        print("Average eval BSD_psnr:{},BSD_ssim:{}".format(BSD_val_psnr, BSD_val_ssim))
        print("Average eval BSD_mse:{},BSD_npcc:{}".format(BSD_val_mse, BSD_val_npcc))
        # ws.cell(row=1, column=5).value = "BSD_mse"
        # ws.cell(row=1, column=6).value = "BSD_psnr"
        # ws.cell(row=1, column=7).value = "BSD_npcc"
        # ws.cell(row=1, column=8).value = "BSD_ssim"
        # ws.cell(row=s, column=5).value = BSD_val_mse.item()
        # ws.cell(row=s, column=6).value = BSD_val_psnr
        # ws.cell(row=s, column=7).value = BSD_val_npcc.item()
        # ws.cell(row=s, column=8).value = BSD_val_ssim.item()

        # 验证celeb数据集
        celeb_psnr_list_ = []
        celeb_ssim_list_ = []
        celeb_mse_list_ = []
        celeb_npcc_list_ = []
        for batch_id, val_data in enumerate(celeb_val_loader):
            with torch.no_grad():
                imgs, label = val_data
                imgs = imgs.to(device)
                label = label.to(device)
                output = swin_model(imgs)
                # output = torch.where(output < 1, output, b)

            # Calculate  eval average psnr, ssim

            celeb_psnr_list_.append(to_psnr(output, label))
            celeb_ssim_list_.append(to_ssim(output, label))
            celeb_mse_list_.append(to_mseloss(output, label))
            celeb_npcc_list_.append(to_pearson(output, label))

        celeb_val_psnr = sum(celeb_psnr_list_) / len(celeb_psnr_list_)
        celeb_val_ssim = sum(celeb_ssim_list_) / len(celeb_ssim_list_)
        celeb_val_mse = sum(celeb_mse_list_) / len(celeb_mse_list_)
        celeb_val_npcc = sum(celeb_npcc_list_) / len(celeb_npcc_list_)
        print("Average eval celeb_psnr:{},celeb_ssim:{}".format(celeb_val_psnr, celeb_val_ssim))
        print("Average eval celeb_mse:{},celeb_npcc:{}".format(celeb_val_mse, celeb_val_npcc))
        # ws.cell(row=1, column=9).value = "celeb_mse"
        # ws.cell(row=1, column=10).value = "celeb_psnr"
        # ws.cell(row=1, column=11).value = "celeb_npcc"
        # ws.cell(row=1, column=12).value = "celeb_ssim"
        # ws.cell(row=s, column=9).value = celeb_val_mse.item()
        # ws.cell(row=s, column=10).value = celeb_val_psnr
        # ws.cell(row=s, column=11).value = celeb_val_npcc.item()
        # ws.cell(row=s, column=12).value = celeb_val_ssim.item()

        # 验证DIV数据集
        DIV_psnr_list_ = []
        DIV_ssim_list_ = []
        DIV_mse_list_ = []
        DIV_npcc_list_ = []
        for batch_id, val_data in enumerate(DIV_val_loader):
            with torch.no_grad():
                imgs, label = val_data
                imgs = imgs.to(device)
                label = label.to(device)
                output = swin_model(imgs)
                # output = torch.where(output < 1, output, b)

            # Calculate  eval average psnr, ssim

            DIV_psnr_list_.append(to_psnr(output, label))
            DIV_ssim_list_.append(to_ssim(output, label))
            DIV_mse_list_.append(to_mseloss(output, label))
            DIV_npcc_list_.append(to_pearson(output, label))

        DIV_val_psnr = sum(DIV_psnr_list_) / len(DIV_psnr_list_)
        DIV_val_ssim = sum(DIV_ssim_list_) / len(DIV_ssim_list_)
        DIV_val_mse = sum(DIV_mse_list_) / len(DIV_mse_list_)
        DIV_val_npcc = sum(DIV_npcc_list_) / len(DIV_npcc_list_)
        print("Average eval DIV_psnr:{},DIV_ssim:{}".format(DIV_val_psnr, DIV_val_ssim))
        print("Average eval DIV_mse:{},DIV_npcc:{}".format(DIV_val_mse, DIV_val_npcc))
        # ws.cell(row=1, column=13).value = "DIV_mse"
        # ws.cell(row=1, column=14).value = "DIV_psnr"
        # ws.cell(row=1, column=15).value = "DIV_npcc"
        # ws.cell(row=1, column=16).value = "DIV_ssim"
        # ws.cell(row=s, column=13).value = DIV_val_mse.item()
        # ws.cell(row=s, column=14).value = DIV_val_psnr
        # ws.cell(row=s, column=15).value = DIV_val_npcc.item()
        # ws.cell(row=s, column=16).value = DIV_val_ssim.item()

        # 验证flickr数据集
        flickr_psnr_list_ = []
        flickr_ssim_list_ = []
        flickr_mse_list_ = []
        flickr_npcc_list_ = []
        for batch_id, val_data in enumerate(flickr_val_loader):
            with torch.no_grad():
                imgs, label = val_data
                imgs = imgs.to(device)
                label = label.to(device)
                output = swin_model(imgs)
                # output = torch.where(output < 1, output, b)

            # Calculate  eval average psnr, ssim

            flickr_psnr_list_.append(to_psnr(output, label))
            flickr_ssim_list_.append(to_ssim(output, label))
            flickr_mse_list_.append(to_mseloss(output, label))
            flickr_npcc_list_.append(to_pearson(output, label))

        flickr_val_psnr = sum(flickr_psnr_list_) / len(flickr_psnr_list_)
        flickr_val_ssim = sum(flickr_ssim_list_) / len(flickr_ssim_list_)
        flickr_val_mse = sum(flickr_mse_list_) / len(flickr_mse_list_)
        flickr_val_npcc = sum(flickr_npcc_list_) / len(flickr_npcc_list_)
        print("Average eval flickr_psnr:{},flickr_ssim:{}".format(flickr_val_psnr, flickr_val_ssim))
        print("Average eval flickr_mse:{},flickr_npcc:{}".format(flickr_val_mse, flickr_val_npcc))
        # ws.cell(row=1, column=17).value = "flickr_mse"
        # ws.cell(row=1, column=18).value = "flickr_psnr"
        # ws.cell(row=1, column=19).value = "flickr_npcc"
        # ws.cell(row=1, column=20).value = "flickr_ssim"
        # ws.cell(row=s, column=17).value = flickr_val_mse.item()
        # ws.cell(row=s, column=18).value = flickr_val_psnr
        # ws.cell(row=s, column=19).value = flickr_val_npcc.item()
        # ws.cell(row=s, column=20).value = flickr_val_ssim.item()

        # 验证imagenet数据集
        WED_psnr_list_ = []
        WED_ssim_list_ = []
        WED_mse_list_ = []
        WED_npcc_list_ = []
        for batch_id, val_data in enumerate(WED_val_loader):
            with torch.no_grad():
                imgs, label = val_data
                imgs = imgs.to(device)
                label = label.to(device)
                output = swin_model(imgs)
                # output = torch.where(output < 1, output, b)

            # Calculate  eval average psnr, ssim

            WED_psnr_list_.append(to_psnr(output, label))
            WED_ssim_list_.append(to_ssim(output, label))
            WED_mse_list_.append(to_mseloss(output, label))
            WED_npcc_list_.append(to_pearson(output, label))
        #
        # WED_val_psnr = sum(WED_psnr_list_) / len(WED_psnr_list_)
        # WED_val_ssim = sum(WED_ssim_list_) / len(WED_ssim_list_)
        # WED_val_mse = sum(WED_mse_list_) / len(WED_mse_list_)
        # WED_val_npcc = sum(WED_npcc_list_) / len(WED_npcc_list_)
        # print("Average eval WED_psnr:{},WED_ssim:{}".format(WED_val_psnr, WED_val_ssim))
        # print("Average eval WED_mse:{},WED_npcc:{}".format(WED_val_mse, WED_val_npcc))
        # # ws.cell(row=1, column=21).value = "WED_mse"
        # # ws.cell(row=1, column=22).value = "WED_psnr"
        # # ws.cell(row=1, column=23).value = "WED_npcc"
        # # ws.cell(row=1, column=24).value = "WED_ssim"
        # # ws.cell(row=s, column=21).value = WED_val_mse.item()
        # # ws.cell(row=s, column=22).value = WED_val_psnr
        # # ws.cell(row=s, column=23).value = WED_val_npcc.item()
        # # ws.cell(row=s, column=24).value = WED_val_ssim.item()
        # # wb.save('epoch_d=f.xlsx')
        # # empty_cache()

        if BSD_val_mse <= old_mse:  # and val_ssim <= old_ssim:
            #torch.save(swin_model.state_dict(), 'parameter/swin_noconv_{}'.format(args.coherence))
            old_mse = BSD_val_mse
            # old_ssim = val_ssim
            print("权重更新完成")
        # if i == 100:
        #     torch.save(RDR_model.state_dict(), 'weight_parameter_gr36{}'.format(i))


if __name__ == '__main__':
    co = ['1.4f']
    for i in co:
        print(i)
        parser = argparse.ArgumentParser()
        parser.add_argument('--coherence', type=str, default='{}'.format(i))
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--val-batch', type=int, default=16)
        parser.add_argument('--epochs', type=int, default=80)
        parser.add_argument('--lr', type=float, default=0.001)
        opt = parser.parse_args()
        main(opt)

# tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
# tb_writer.add_scalar(tags[0], train_loss, epoch)
# tb_writer.add_scalar(tags[1], train_acc, epoch)
# tb_writer.add_scalar(tags[2], val_loss, epoch)
# tb_writer.add_scalar(tags[3], val_acc, epoch)
# tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
#
# torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
