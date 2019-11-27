import argparse
import os
import shutil
import time
from importlib import import_module
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.optim
import dataset
import utils

tasks = ['train', 'test', 'predict']
todo = [['train()', 'test()'],
        ['test()'],
        ['predict()']]


def train():
    global model, net, dir_model, dir_root, initial_epoch, device

    print('#'*5, 'training: ')

    model.train()
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=net.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=net.lr_milestone, gamma=net.lr_gamma)

    train_dataset = dataset.VideoDataset(lres_npz_dir=net.lres_npz_dir,
                                         sample_first_percentage=net.train_first_percentage,
                                         scale=net.scale,
                                         lres_patch_size=net.patch_size, lres_patch_stride=net.patch_stride,
                                         nframes=net.nframes,
                                         augmentation=net.aug,
                                         hres_npz_dir=net.hres_npz_dir,
                                         channel=net.channel
                                         )
    print('Dataset: number of total sample {}'.format(len(train_dataset.keys)))

    best_loss = -1

    def get_dataloader(net, train_dataset):
        try:
            net.train_sample_reduce
        except NameError:
            pass
        else:
            train_dataset.random_sample(net.train_sample_reduce)
            print('Dataset: random sample {} of all, number of sample {}'.format(
                net.train_sample_reduce, len(train_dataset.keys)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=net.batch_size, num_workers=net.num_workers,
                                                   shuffle=True)
        return train_loader

    def model_save(model2save, current_epoch):
        assert isinstance(model2save, torch.nn.Module)
        assert not isinstance(model2save, torch.nn.DataParallel)
        torch.save(model2save.state_dict(), os.path.join(
            dir_model, 'model_{:03d}.pth'.format(current_epoch)))

    def model_save_best(current_epoch):
        shutil.copy(
            os.path.join(dir_model, 'model_{:03d}.pth'.format(current_epoch)),
            os.path.join(dir_model, '..', 'model_best.pth'))

    for epoch in range(initial_epoch, net.n_epoch):
        train_loader = get_dataloader(net, train_dataset)
        epoch += 1
        scheduler.step(epoch)
        epoch_loss = 0
        n_count, n_total = 1, len(train_loader)
        print('Epoch {} start. lr = {:8f}'.format(
            epoch, scheduler.get_lr()[0]))
        t1 = time.time()
        for batch_input in train_loader:
            lres, hres = batch_input[0], batch_input[1]
            lres = lres.view(lres.shape[0], net.nframes, lres.shape[1]//net.nframes, lres.shape[2], lres.shape[3])
            lres = lres.permute(0,2,1,3,4).contiguous()
            lres, hres = lres.to(device), hres.to(device)
            lres /= net.value_in_max
            hres /= net.value_out_max

            # predict
            sres = model(lres)
            sres = torch.clamp(sres, 0, 1)

            # loss
            loss = net.criterion(sres, hres)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print('\rTraining {}/{} loss = {:8.6f}'.format(n_count,
                                                           n_total, loss.item()), end='')
            n_count += 1
        t2 = time.time()
        print('\rEpoch {} done. Loss = {:8.6f}, Time = {:4.2f}s, Loss(x) = {:8.2f}'.format(
            epoch, epoch_loss/n_total, t2 - t1, epoch_loss/n_total*net.value_out_max))

        if isinstance(model, torch.nn.DataParallel):
            model_save(model.module, epoch)
        else:
            model_save(model, epoch)

        if best_loss == -1:
            best_loss = epoch_loss
        else:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                model_save_best(epoch)
                with open(os.path.join(dir_root, 'model_better.txt'), 'a') as fp:
                    fp.write(str(epoch)+'\n')


def test():
    global model, net, dir_model, device

    print('#'*5, 'testing: latest')

    latest = utils.find_last_checkpoint(save_dir=dir_model)
    pth_data = torch.load(os.path.join(
        dir_model, 'model_{:03d}.pth'.format(latest)))
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(pth_data)
    else:
        model.load_state_dict(pth_data)

    test_model()

    print('#'*5, 'testing: best')

    pth_data = torch.load(os.path.join(dir_model, '..', 'model_best.pth'))
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(pth_data)
    else:
        model.load_state_dict(pth_data)

    test_model()


def test_model():
    global model, net, device

    model.eval()
    torch.backends.cudnn.benchmark = False

    test_dataset = dataset.VideoDataset(lres_npz_dir=net.lres_npz_dir,
                                        sample_first_percentage=net.test_first_percentage,
                                        scale=net.scale,
                                        nframes=net.nframes,
                                        augmentation=False,
                                        hres_npz_dir=net.hres_npz_dir,
                                        channel=net.channel
                                        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=net.num_workers_test)

    n_count, n_total = 1, len(test_loader)
    test_loss = 0
    t1 = time.time()
    with torch.no_grad():
        for batch_input in test_loader:
            lres, hres = batch_input[0], batch_input[1]
            lres = lres.view(lres.shape[0], net.nframes, lres.shape[1]//net.nframes, lres.shape[2], lres.shape[3])
            lres = lres.permute(0,2,1,3,4).contiguous()
            lres, hres = lres.to(device), hres.to(device)
            lres = lres/net.value_in_max
            hres = hres/net.value_out_max

            sres = model(lres)
            sres = torch.clamp(sres, 0, 1)

            loss = net.criterion(sres, hres)

            test_loss += loss.item()

            print('\rTesting {}/{}'.format(n_count, n_total), end='')
            n_count += 1
    t2 = time.time()
    print('\rTest done. Loss = {:8f}, Time = {:6f}s, Loss(x) = {:8.2f}'.format(
        test_loss/n_total, t2-t1, test_loss/n_total*net.value_out_max))


def predict():
    global model, net, dir_model, dir_result, device
    if os.path.exists(os.path.join(dir_result, 'latest')):
        shutil.rmtree(os.path.join(dir_result, 'latest'))
    if os.path.exists(os.path.join(dir_result, 'best')):
        shutil.rmtree(os.path.join(dir_result, 'best'))

    print('#'*5, 'testing: latest')

    latest = utils.find_last_checkpoint(save_dir=dir_model)
    pth_data = torch.load(os.path.join(
        dir_model, 'model_{:03d}.pth'.format(latest)))
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(pth_data)
    else:
        model.load_state_dict(pth_data)

    predict_model('latest')

    
    print('#'*5, 'testing: best')

    pth_data = torch.load(os.path.join(dir_model, '..', 'model_best.pth'))
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(pth_data)
    else:
        model.load_state_dict(pth_data)

    predict_model('best')


def predict_model(result_subdir):
    global model, net, dir_result, device

    import threading

    class WorkerSaveNpz(threading.Thread):
        def __init__(self, tensor:torch.FloatTensor, value_out_max:int, path_to_save:str):
            super().__init__()
            self.value_out_max = value_out_max
            self.tensor = tensor
            self.path_to_save = path_to_save
        
        def run(self):
            assert self.tensor.shape[0] == 1
            import numpy as np
            dirname = os.path.dirname(self.path_to_save)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            sres:np.ndarray = self.tensor[0,:,:,:].numpy()
            sres = sres * self.value_out_max
            if self.value_out_max == 255:
                dtype = 'uint8'
            elif self.value_out_max == 1023:
                dtype = 'uint16'
            sres_ch = dataset.sres_to_ch(sres.round().astype(dtype))
            np.savez_compressed(self.path_to_save, y=sres_ch[0], u=sres_ch[1], v=sres_ch[2])

        
    def x8_transform(x:torch.FloatTensor, id:int):
        if id == 0:
            return x
        elif id == 1:
            return x.rot90(1, [-2,-1])
        elif id == 2:
            return x.rot90(2, [-2,-1])
        elif id == 3:
            return x.rot90(3, [-2,-1])
        elif id == 4:
            return x.flip(-1)
        elif id == 5:
            return x.flip(-1).rot90(1, [-2, -1])
        elif id == 6:
            return x.flip(-1).rot90(2, [-2, -1])
        elif id == 7:
            return x.flip(-1).rot90(3, [-2, -1])
        
    def x8_inv_transform(x:torch.FloatTensor, id:int):
        if id == 0:
            return x
        elif id == 1:
            return x.rot90(3, [-2,-1])
        elif id == 2:
            return x.rot90(2, [-2,-1])
        elif id == 3:
            return x.rot90(1, [-2,-1])
        elif id == 4:
            return x.flip(-1)
        elif id == 5:
            return x.rot90(3, [-2, -1]).flip(-1)
        elif id == 6:
            return x.rot90(2, [-2, -1]).flip(-1)
        elif id == 7:
            return x.rot90(1, [-2, -1]).flip(-1)
        

    x8 = input('if x8 forward ? (y/N)')
    if 'y' in x8.lower():
        x8 = True
        result_subdir = result_subdir + '_x8'
    else:
        x8 = False

    workers = []
    worker_limit = 15

    model.eval()
    torch.backends.cudnn.benchmark = False

    test_dataset = dataset.VideoDataset(lres_npz_dir=net.predict_npz_dir,
                                        scale=net.scale,
                                        nframes=net.nframes,
                                        augmentation=False,
                                        channel=net.channel
                                        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=net.num_workers_test)

    n_count, n_total = 1, len(test_loader)
    t1 = time.time()
    with torch.no_grad():
        for batch_input in test_loader:
            lres:torch.FloatTensor = batch_input.to(device)
            lres = lres.view(lres.shape[0], net.nframes, lres.shape[1]//net.nframes, lres.shape[2], lres.shape[3])
            lres = lres.permute(0,2,1,3,4).contiguous()
            lres = lres/net.value_in_max

            if x8:
                sres_x8 = []
                for i in range(8):
                    lres_x8 = x8_transform(lres, i)

                    sres = model(lres_x8)
                    sres.detach()
                    sres = torch.clamp(sres, 0, 1)
                    sres_x8.append(x8_inv_transform(sres, i))
                sres = sres_x8[0]
                for i in range(1,8):
                    sres += sres_x8[i]
                sres /= 8
                sres = torch.clamp(sres, 0, 1)
            else:
                sres = model(lres)
                sres.detach()
                sres = torch.clamp(sres, 0, 1)
    
            vidx, fidx, _, _= test_dataset.keys[n_count-1]
            t = WorkerSaveNpz(sres.cpu(), net.value_out_max,
                os.path.join(dir_result, result_subdir, test_dataset.videos[vidx], test_dataset.frames[vidx][fidx]))
            t.start()
            workers.append(t)

            print('\rPredicting {}/{}, saving to {}'.format(n_count, n_total, t.path_to_save), end='')
            n_count += 1

            while len(workers) > worker_limit:
                workers[0].join()
                workers.pop(0)

    while len(workers) > 0:
        workers[0].join()
        workers.pop(0)

    t2 = time.time()
    print('\nPredict done. Time = {:.2f}s'.format(t2-t1))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train, test network, or predict')
    parser.add_argument('-n', '--net', type=str, required=True,
                        help='filename of neural network definition file, with ext ".py", '
                             'must be in the same folder with main.py')
    parser.add_argument('-t', '--type', type=str, required=True,
                        help='type of running, valid values in ["train", "test", "predict"]')
    parser.add_argument('-g', '--ngpus', type=int, default=1,
                        help='gpu is default, use this for multi gpu scenario')
    parser.add_argument('-l', '--load_pth', type=str,
                        help='path of pth file to load')
    args = parser.parse_args()

    assert args.net[-3:] == '.py' and os.path.isfile(args.net)
    assert args.type in tasks

    if args.load_pth is not None:
        assert args.load_pth[-4:] == '.pth' and os.path.isfile(args.load_pth)

    assert torch.cuda.is_available()

    model_name = os.path.split(args.net[:-3])[1]
    net = import_module(model_name)

    model = net.Net()
    device = 'cuda:0'
    if args.ngpus > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.ngpus)))

    dir_root = os.path.join('../exp', model_name)
    dir_result = os.path.join('../exp', model_name, 'result')
    dir_model = os.path.join('../exp', model_name, 'model')

    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    shutil.copy(args.net, os.path.join(dir_root, model_name + '.py'))

    initial_epoch = 0
    pth_data = None
    if args.load_pth is not None:
        pth_data = torch.load(args.load_pth)
    else:
        initial_epoch = utils.find_last_checkpoint(save_dir=dir_model)
        if initial_epoch > 0:
            print('loading epoch {:03d}'.format(initial_epoch))
            pth_data = torch.load(os.path.join(
                dir_model, 'model_{:03d}.pth'.format(initial_epoch)))
    if pth_data is not None:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(pth_data)
        else:
            model.load_state_dict(pth_data)

    model = model.to(device)

    for i in todo[tasks.index(args.type)]:
        exec(i)
