import argparse
import os
import sys
import os.path as osp
import glob
import numpy as np
import torch
from utils import Logger
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# from model.rcan import RCAN
from model.model import Net
from dataloder import DatasetLoaderWithHR, DatasetLoader
from utils import save_checkpoint, train


# Loss function begin <<<
class CBLoss(nn.Module):
    """Charbonnier Loss (L1) + CosineSimilarity"""

    def __init__(self, eps=1e-6, loss_lambda=2):
        super().__init__()
        self.eps = eps
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=eps)
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        cosine_term = (1 - self.similarity(x, y)).mean()

        return loss + self.loss_lambda * cosine_term

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        loss = self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        return loss

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class TCBLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tv_loss = TVLoss()
        self.cb_loss = CBLoss()

    def forward(self, output, target):
        return 1e-5 * self.tv_loss(output) + self.cb_loss(output, target)     
# Loss function end >>>

def main(arg):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log_naic_round2.txt'))
    print("====>> Read file list")
    file_name = sorted(os.listdir(arg.VIDEO4K_LR))
    lr_list = []
    hr_list = []
    for fi in file_name:
        lr_tmp = sorted(glob.glob(arg.VIDEO4K_LR + '/' + fi + '/*.png'))
        lr_list.extend(lr_tmp)
        hr_tmp = sorted(glob.glob(arg.VIDEO4K_HR + '/' + fi + '/*.png'))
        if len(hr_tmp) != 100:
            print(fi)
        hr_list.extend(hr_tmp)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(len(lr_list))
    print(len(hr_list))
    cudnn.benchmark = True

    print("===> Loading datasets")
    data_set = DatasetLoader(lr_list, hr_list, arg.patch_size, arg.scale)
    train_loader = DataLoader(data_set, batch_size=arg.batch_size,
                              num_workers=arg.workers, shuffle=True, pin_memory=True)

    print("===> Building model")
    device_ids = list(range(args.ngpus))
    model = Net(arg)
    criterion = TCBLoss()

    print("===> Setting GPU")
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    criterion = criterion.cuda()

    # optionally ckp from a checkpoint
    if arg.ckp:
        if os.path.isfile(args.ckp):
            print("=> loading checkpoint '{}'".format(arg.ckp))
            checkpoint = torch.load(arg.ckp)
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            # load params
            model.load_state_dict(new_state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.ckp))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=arg.lr, weight_decay=arg.weight_decay, betas=(0.9, 0.999), eps=1e-08)

    print("===> Training")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)
        train(train_loader, optimizer, model, criterion, epoch, arg, len(hr_list), )
        save_checkpoint(model, epoch)


def adjust_lr(opt, epoch):
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in [40, 80, 120]:
        args.lr *= 0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    # model parameter
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--step_batch_size', default=1, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('-g', '--ngpus', type=int, default=1,
                        help='gpu is default, use this for multi gpu scenario')

    # path
    parser.add_argument('--VIDEO4K_LR', type=str, metavar='PATH',
                        default='/root/proj/NAIC/dataset/round2/img_lr')
    parser.add_argument('--VIDEO4K_HR', type=str, metavar='PATH',
                        default='/root/proj/NAIC/dataset/img_hr')

    # check point
    parser.add_argument("--ckp", default='ckp/model_epoch_100.pth', type=str)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument("--logs_dir", default='log/', type=str)

    args = parser.parse_args()
    main(args)
