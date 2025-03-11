import numpy as np
import torch
import torch.nn.functional as F
import os
from torch import distributed
import torchvision.datasets as datasets
from PIL import Image


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m/s, (1 - m)/s)
    return image_tensor


def tiny_clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance, using tiny-imagenet normalization
    '''
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def cifar_100_clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance, using tiny-imagenet normalization
    '''
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def cifar_10_clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance, using tiny-imagenet normalization
    '''
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2471, 0.2435, 0.2616])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def tiny_denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input, using tiny-imagenet normalization
    '''

    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def cifar_100_denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input, using tiny-imagenet normalization
    '''

    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def cifar_10_denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input, using tiny-imagenet normalization
    '''

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2471, 0.2435, 0.2616])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def in21k_save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = "{}/new{:05d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:03d}.jpg".format(class_id, ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def validate(input, target, model):
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

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


class ViT_BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        B, N, C = input[0].shape

        mean = torch.mean(input[0], dim=[0, 1])
        var = torch.var(input[0], dim=[0, 1], unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)


        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


# modified from Alibaba-ImageNet21K/src_files/models/utils/factory.py
def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')

    Flag = False
    if 'state_dict' in state:
        ### resume from a model trained with nn.DataParallel
        state = state['state_dict']
        Flag = True

    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]

        if Flag:
            key = 'module.' + key

        if key in state:
            ip = state[key]
        # if key in state['state_dict']:
        #     ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


def entropy(tensor, dim=-1, eps=1e-10):
    """
    Calculate the entropy of a tensor after applying the softmax operator.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the entropy.
        eps (float): Small constant to prevent division by zero.

    Returns:
        torch.Tensor: Entropy of the input tensor.
    """
    # Apply softmax to the input tensor
    softmax_probs = F.softmax(tensor, dim=dim)

    # Clip the probabilities to avoid log(0)
    softmax_probs = torch.clamp(softmax_probs, min=eps, max=1.0)
    softmax_probs_log = torch.log(softmax_probs)

    # Compute the negative entropy
    entropy_values = -torch.mean(torch.mean(softmax_probs * softmax_probs_log, dim=dim))

    return entropy_values


def entropy_with_logsoftmax(tensor, dim=-1):
    """
    Calculate the entropy of a tensor after applying the log softmax operator.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the entropy.

    Returns:
        torch.Tensor: Entropy of the input tensor.
    """
    # Apply log softmax to the input tensor
    log_softmax_probs = F.log_softmax(tensor, dim=dim)
    softmax_probs = torch.exp(log_softmax_probs)
    # Compute the negative entropy
    entropy_values = -torch.mean(torch.mean(softmax_probs * log_softmax_probs, dim=dim))

    return entropy_values


def entropy_with_logsoftmax_non_saturating(tensor, dim=-1):
    """
    Calculate the entropy of a tensor after applying the log softmax operator.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the entropy.

    Returns:
        torch.Tensor: Entropy of the input tensor.
    """
    # Apply log softmax to the input tensor
    log_softmax_probs = F.log_softmax(tensor, dim=dim)
    softmax_probs = torch.exp(log_softmax_probs)
    log_softmax_probs_ = F.log_softmax(1 - tensor, dim=dim)
    # Compute the negative entropy
    entropy_values = torch.mean(torch.mean(softmax_probs * log_softmax_probs_, dim=dim))

    return entropy_values


class ClassFolder(datasets.DatasetFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 class_idx=0):
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ClassFolder, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        # Override
        self.classes, self.class_to_idx = self.find_subclasses(class_idx=class_idx)
        self.nclass = 1
        self.samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions,
                                                    is_valid_file)


        self.targets = [s[1] for s in self.samples]

    def find_subclasses(self, class_idx=0):
        """Finds the class folders in a dataset.
        """
        classes = []
        class_indices = [class_idx]
        for i in class_indices:
            classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == 1

        return classes, class_to_idx

    def __getitem__(self, index):
        path = self.samples[index][0]
        sample = self.loader(path)

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

