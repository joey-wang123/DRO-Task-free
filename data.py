import os
import sys
import pdb
import torch
import numpy as np
import pickle as pkl
from PIL import Image
from random import shuffle

from torchvision import datasets, transforms


""" Template Dataset with Labels """
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y

        # this was to store the inverse permutation in permuted_mnist
        # so that we could 'unscramble' samples and plot them
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if type(x) != torch.Tensor:
            # mini_imagenet
            # we assume it's a path --> load from file
            x = self.transform(Image.open(x).convert('RGB'))
            y = torch.Tensor(1).fill_(y).long().squeeze()
        else:
            x = x.float() / 255.
            y = y.long()


        # for some reason mnist does better \in [0,1] than [-1, 1]
        if self.source == 'mnist':
            return x, y
        else:
            return (x - 0.5) * 2, y


""" Template Dataset for Continual Learning """
class CLDataLoader(object):
    def __init__(self, datasets_per_task, args, train=True):
        bs = args.batch_size if train else 64
        self.datasets = datasets_per_task

        # concat_list = []
        # for x in self.datasets:
        #     concat_list.append(x)
        #     print('loader x', x)
        self.loaders = [
                torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=train, num_workers=0)
                for x in self.datasets ]

        # print('len(concat_list)', len(concat_list))
        # print('len(self.loaders)', len(self.loaders))

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)




""" Split CIFAR10 into 5 tasks {{0,1}, ... {8,9}} """
def get_split_cifar10(args):
    # assert args.n_tasks in [5, 10], 'SplitCifar only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'
    args.n_tasks   = 5
    args.n_classes = 10
    args.buffer_size = args.n_tasks * args.mem_size * 2
    args.use_conv = True
    args.n_classes_per_task = 2
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'
    # because data is between [-1,1]:
    assert args.output_loss is not 'bernouilli'
    if args.output_loss == None:
        #TODO(multinomial is broken)
        #args.output_loss = 'multinomial'
        args.output_loss = 'mse'
        print('\nsetting output loss to MSE')


    # fetch MNIST
    train = datasets.CIFAR10('Data/', train=True,  download=True)
    test  = datasets.CIFAR10('Data/', train=False, download=True)

    try:
        train_x, train_y = train.data, train.targets
        test_x, test_y = test.data, test.targets
    except:
        train_x, train_y = train.train_data, train.train_labels
        test_x,  test_y  = test.test_data,   test.test_labels

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]

    out_test = [
        (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            np.stack([elem[i] for elem in out_train]) for i in [0,1] ]

    test_x,  test_y  = [
            np.stack([elem[i] for elem in out_test]) for i in [0,1] ]

    train_x = torch.Tensor(train_x).permute(0, 3, 1, 2).contiguous()
    test_x  = torch.Tensor(test_x).permute(0, 3, 1, 2).contiguous()

    train_y = torch.Tensor(train_y)
    test_y  = torch.Tensor(test_y)

    # get indices of class split
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = [0] + [x + 1 for x in sorted(train_idx)]

    test_idx  = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx  = [0] + [x + 1 for x in sorted(test_idx)]

    train_ds, test_ds = [], []
    skip = 10 // 5 #args.n_tasks
    for i in range(0, 10, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i],  test_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    train_ds, val_ds = make_valid_from_train(train_ds)

    train_ds = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), train_ds)
    val_ds   = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), val_ds)
    test_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), test_ds)

    return train_ds, val_ds, test_ds



def get_split_cifar100(args):
    # assert args.n_tasks in [5, 10], 'SplitCifar only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'
    args.n_tasks   = 20
    args.n_classes = 100
    args.buffer_size = args.n_tasks * args.mem_size * 5
    args.use_conv = True
    args.n_classes_per_task = 5
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'
    # because data is between [-1,1]:
    assert args.output_loss is not 'bernouilli'
    if args.output_loss == None:
        #TODO(multinomial is broken)
        #args.output_loss = 'multinomial'
        args.output_loss = 'mse'
        print('\nsetting output loss to MSE')


    # fetch MNIST
    train = datasets.CIFAR100('Data/', train=True,  download=True)
    test  = datasets.CIFAR100('Data/', train=False, download=True)

    try:
        train_x, train_y = train.data, train.targets
        test_x, test_y = test.data, test.targets
    except:
        train_x, train_y = train.train_data, train.train_labels
        test_x,  test_y  = test.test_data,   test.test_labels

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]

    out_test = [
        (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            np.stack([elem[i] for elem in out_train]) for i in [0,1] ]

    test_x,  test_y  = [
            np.stack([elem[i] for elem in out_test]) for i in [0,1] ]

    train_x = torch.Tensor(train_x).permute(0, 3, 1, 2).contiguous()
    test_x  = torch.Tensor(test_x).permute(0, 3, 1, 2).contiguous()

    train_y = torch.Tensor(train_y)
    test_y  = torch.Tensor(test_y)

    # get indices of class split
    train_idx = [((train_y + i) % 100).argmax() for i in range(100)]
    train_idx = [0] + [x + 1 for x in sorted(train_idx)]

    test_idx  = [((test_y + i) % 100).argmax() for i in range(100)]
    test_idx  = [0] + [x + 1 for x in sorted(test_idx)]

    train_ds, test_ds = [], []
    skip = 100 // 20 #args.n_tasks
    for i in range(0, 100, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i],  test_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    train_ds, val_ds = make_valid_from_train(train_ds)

    train_ds = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), train_ds)
    val_ds   = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), val_ds)
    test_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), test_ds)

    return train_ds, val_ds, test_ds




def get_miniimagenet(args):

    print('loading miniimagenet dataset')
    ROOT_PATH = '/Data/Miniimagenet/'

    args.use_conv = True
    args.n_tasks   = 20
    args.n_classes = 100
    args.n_classes_per_task = 5
    args.input_size = (3, 84, 84)
    label2id = {}

    def get_data(setname):
        ds_dir = os.path.join(ROOT_PATH, setname)
        label_dirs = os.listdir(ds_dir)
        data, labels = [], []

        for label in label_dirs:
            label_dir = os.path.join(ds_dir, label)
            for image_file in os.listdir(label_dir):
                data.append(os.path.join(label_dir, image_file))
                if label not in label2id:
                    label_id = len(label2id)
                    label2id[label] = label_id
                label_id = label2id[label]
                labels.append(label_id)
        return data, labels

    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
    ])

    train_data, train_label = get_data('meta_train')
    valid_data, valid_label = get_data('meta_val')
    test_data,  test_label  = get_data('meta_test')

    # total of 60k examples for training, the rest for testing
    all_data  = np.array(train_data  + valid_data  + test_data)
    all_label = np.array(train_label + valid_label + test_label)


    train_ds, test_ds = [], []
    current_train, current_test = None, None

    cat = lambda x, y: np.concatenate((x, y), axis=0)

    for i in range(args.n_classes):
        class_indices = np.argwhere(all_label == i).reshape(-1)
        class_data  = all_data[class_indices]
        class_label = all_label[class_indices]
        split = int(0.8 * class_data.shape[0])

        data_train, data_test = class_data[:split], class_data[split:]
        label_train, label_test = class_label[:split], class_label[split:]

        if current_train is None:
            current_train, current_test = (data_train, label_train), (data_test, label_test)
        else:
            current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
            current_test  = cat(current_test[0],  data_test),  cat(current_test[1],  label_test)

        if i % args.n_classes_per_task == (args.n_classes_per_task  - 1):
            train_ds += [current_train]
            test_ds  += [current_test]
            current_train, current_test = None, None


    # build masks
    masks = []
    task_ids = [None for _ in range(20)]
    for task, task_data in enumerate(train_ds):
        labels = np.unique(task_data[1]) #task_data[1].unique().long()
        assert labels.shape[0] == args.n_classes_per_task
        mask = torch.zeros(args.n_classes).cuda()
        mask[labels] = 1
        masks += [mask]
        task_ids[task] = labels

    task_ids = torch.from_numpy(np.stack(task_ids)).cuda().long()
    print('task_ids', task_ids)


    train_ds, val_ds = make_valid_from_train(train_ds)
    train_ds = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids, 'transform':transform}), train_ds, masks)
    val_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'cifar100', 'mask': y, 'task_ids': task_ids, 'transform': transform}), val_ds, masks)
    test_ds  = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids, 'transform':transform}), test_ds, masks)


    return train_ds, val_ds, test_ds


def make_valid_from_train(dataset, cut=0.95):
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t, y_t = task_ds

        # shuffle before splitting
        perm = torch.randperm(len(x_t))
        x_t, y_t = x_t[perm], y_t[perm]

        split = int(len(x_t) * cut)
        x_tr, y_tr   = x_t[:split], y_t[:split]
        x_val, y_val = x_t[split:], y_t[split:]

        tr_ds  += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds





class IIDDataset(torch.utils.data.Dataset):
    def __init__(self, data_loaders, seed=0):
        self.data_loader = data_loaders
        self.idx = []
        for task_id in range(len(data_loaders)):
            for i in range(len(data_loaders[task_id].dataset)):
                self.idx.append((task_id, i))
        random.Random(seed).shuffle(self.idx)

    def __getitem__(self, idx):
        task_id, instance_id = self.idx[idx]
        return self.data_loader[task_id].dataset.__getitem__(instance_id)

    def __len__(self):
        return len(self.idx)