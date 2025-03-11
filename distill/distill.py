'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import argparse
import collections
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from utils import *
import non_saturating_loss
import copy


def get_images(args, model_teacher, model_student, hook_for_display, ipc_id):
    print("get_images call")
    save_every = args.iteration
    batch_size = args.batch_size

    best_cost = 1e4

    model_teacher_copy = copy.deepcopy(model_teacher)
    model_teacher_copy.eval()
    feature_extractor = torch.nn.Sequential(*list(model_teacher.children())[:-1])
    feature_extractor.eval()

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    targets_all = torch.LongTensor(np.arange(1000))

    for kk in range(0, 1000, batch_size):
        targets = targets_all[kk: min(kk + batch_size, 1000)].to('cuda')

        print('init from original dataset')
        inputs = torch.load('{}/id{:03d}_batch{:03d}.pt'.format(args.init_path, ipc_id, kk // batch_size)).cuda()
        inputs.requires_grad = True
        print(f'inputs size: {inputs.size()}')
        originals = copy.deepcopy(inputs.detach())
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)

        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter , args.jitter

        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer) # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        criterion_l2 = nn.MSELoss()
        adv_criterion = non_saturating_loss.NonSaturatingLoss(args.epsilon)

        for iteration in range(iterations_per_layer):
            lr_scheduler(optimizer, iteration, iteration)

            loss_l2 = torch.tensor(0.0).cuda()

            min_crop = 0.08
            max_crop = 1.0

            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(min_crop, max_crop)),
                transforms.RandomHorizontalFlip(),
            ])
            con = []
            con.append(inputs)
            con.append(originals.detach())
            con = aug_function(torch.cat(con))
            originals_jit = con[originals.size()[0]:]
            inputs_jit = con[:originals.size()[0]]

            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
            if args.init_path is not None:
                originals_jit = torch.roll(originals_jit, shifts=(off1, off2), dims=(2, 3))

            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            loss_l2 = criterion_l2(inputs_jit.reshape(batch_size, -1), originals_jit.reshape(batch_size, -1))

            loss_ce = criterion(outputs, targets)

            rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            loss_aux = args.r_bn * loss_r_bn_feature \
                        + args.l2 * loss_l2

            loss_non_saturating = torch.tensor(0.0).cuda()
            if model_student is not None:
                outputs_student = model_student(inputs_jit)
                _, output_teacher = torch.topk(outputs.detach(), k=1, dim=1)
                output_teacher = output_teacher.squeeze(1)

                mask = torch.ones(output_teacher.size()[0]).cuda()
                for i in range(kk, min(kk + batch_size, 1000)):
                    if output_teacher[i - kk] != targets[i - kk]:
                        mask[i - kk] = 0
                outputs_student = outputs_student * mask.unsqueeze(1)
                loss_non_saturating = args.adv * adv_criterion(outputs_student, targets)

            loss = loss_ce + loss_aux + loss_non_saturating

            if (iteration + 1) % save_every == 0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", loss_ce.item())
                print("student adv criterion", loss_non_saturating.item())
                print("l2 regularization", args.l2 * loss_l2.item())
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            loss.backward()
            optimizer.step()

            inputs.data = clip(inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()
            best_inputs = denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)

        optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()


def main_syn(ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    # teacher
    model_teacher = models.__dict__[args.arch_name](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # student
    if args.student_dir:
        model_student = models.__dict__[args.arch_name](pretrained=False)
        model_student = nn.DataParallel(model_student).cuda()
        ckpt = torch.load(args.student_dir, map_location='cpu')['state_dict']
        model_student.load_state_dict(ckpt)
        model_student.eval()
        for p in model_student.parameters():
            p.requires_grad = False
    else:
        model_student = None

    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    model_verifier = model_verifier.cuda()
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False

    hook_for_display = lambda x,y: validate(x, y, model_verifier)
    get_images(args, model_teacher, model_student, hook_for_display, ipc_id)


def parse_args():
    parser = argparse.ArgumentParser("CUDD")
    parser.add_argument('--exp-name', type=str, default='test', help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--syn-data-path', type=str, default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',  help='whether to store best images')
    parser.add_argument('--ipc-start', default=20, type=int)
    parser.add_argument('--ipc-end', default=50, type=int)
    parser.add_argument('--init-path', type=str, default=None, help='where to init synthetic data')
    parser.add_argument('--batch-size', type=int, default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=4000, help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.25, help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.01, help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--l2', type=float, default=1, help='coefficient for l2 regularization between synthetic and original data')
    parser.add_argument('--adv', type=float, default=1, help='student adv loss on the image')
    parser.add_argument('--epsilon', type=float, default=0, help='epsilon smooth of student adv non saturating')
    parser.add_argument('--soft-target', action='store_true', help='whether to use soft target')
    parser.add_argument('--arch-name', type=str, default='resnet18', help='arch name from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true', help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2', help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--student-dir', type=str, default=None, help='path to pretrained student model dir')
    args = parser.parse_args()
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    return args


if __name__ == '__main__':
    args = parse_args()
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print(f'ipc = {ipc_id}')
        main_syn(ipc_id)
