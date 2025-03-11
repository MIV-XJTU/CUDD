import numpy as np
import torch
import torchvision.datasets as datasets


# keep top k largest values, and smooth others
def keep_top_k(p,k,n_classes=1000): # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes-k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(topk_smooth_p.sum().item(), p.shape[0]), f'{topk_smooth_p.sum().item()} not close to {p.shape[0]}'
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups


class CLImageFolder(datasets.DatasetFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 nclass=1000,
                 step=0,
                 nstep=5,
                 seed=42):
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(CLImageFolder, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        # Override
        self.classes, self.class_to_idx = self.find_subclasses(nclass, step, nstep, seed)
        self.samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions,
                                                    is_valid_file)


        self.targets = [s[1] for s in self.samples]
        print(self.targets)

    def find_subclasses(self, nclass=1000, step=0, nstep=5, seed=42):
        """Finds the class folders in a dataset.
        """
        classes = []
        num_classes_step = nclass // nstep
        np.random.seed(seed)
        class_order = np.random.permutation(nclass).tolist()
        classes_seen = class_order[: (step + 1) * num_classes_step]
        print(f'classes_seen: {classes_seen}')
        class_indices = classes_seen
        for i in class_indices:
            classes.append(self.classes[i])

        class_to_idx = {cls_name: classes_seen[i] for i, cls_name in enumerate(classes)}
        print(f'class_to_idx: {class_to_idx}')
        assert len(classes) == len(classes_seen)

        return classes, class_to_idx

    def __getitem__(self, index):
        path = self.samples[index][0]
        sample = self.loader(path)

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

