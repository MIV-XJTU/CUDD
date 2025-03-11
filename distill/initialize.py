import argparse
import collections
import os
from utils import ClassFolder
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from utils import *


def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.init_path):
            os.mkdir(args.init_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.init_path, class_id)
        place_to_store = '{}/class{:03d}_id{:03d}.jpg'.format(dir_path, class_id, ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def class_loader(args, normalize, c):
    dataloader = torch.utils.data.DataLoader(
                    ClassFolder(args.original_dir, transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]), class_idx=c),
                    batch_size=32, shuffle=True,
                    num_workers=16, pin_memory=True)
    return dataloader


def check_repitition(path, path_all):
    unrepeat = []
    path_unrepeat = []
    for i in range(len(path)):
        if path[i] not in path_all:
            unrepeat.append(i)
            path_unrepeat.append(path[i])
    return unrepeat, path_unrepeat


def filter(model_teacher, model_student, image, target, path, path_all, mode):
    index = []
    path_new = []
    for i in range(image.size()[0]):
        if mode == 'simple':
            _, output_teacher = torch.topk(model_teacher(image), k=1, dim=1)
            output_teacher = output_teacher.squeeze(1)
            if output_teacher[i] == target[i]:
                index.append(i)
                path_new.append(path[i])
        elif mode == 'medium':
            _, output_student = torch.topk(model_student(image), k=1, dim=1)
            output_student = output_student.squeeze(1)
            if output_student[i] != target[i]:
                index.append(i)
                path_new.append(path[i])
        elif mode == 'hard':
            _, output_teacher = torch.topk(model_teacher(image), k=1, dim=1)
            output_teacher = output_teacher.squeeze(1)
            _, output_student = torch.topk(model_student(image), k=1, dim=1)
            output_student = output_student.squeeze(1)
            if output_teacher[i] != target[i] and output_student[i] != target[i]:
                index.append(i)
                path_new.append(path[i])
    image_filtered = image[index]
    path_filtered = path_new

    return image_filtered, path_filtered


def update_data_path(init_data, path_all, image, path):
    for i in range(image.size()[0]):
        init_data.append(image[i].unsqueeze(0))
        path_all.append(path[i])
    return init_data, path_all


def main_syn():
    if not os.path.exists(args.init_path):
        os.makedirs(args.init_path)

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
        print('init without student')

    if args.form_batch: # form data batches
        for ipc in range(args.init_num):
            for start in range(0, 1000, 100):
                ipc_data = []
                for c in range(start, start + 100):
                    class_data = torch.load(os.path.join(args.init_path, 'data_class{:03d}.pt'.format(c)))
                    class_data = class_data[ipc].unsqueeze(0)
                    ipc_data.append(class_data)
                    del class_data
                assert len(ipc_data) == 100

                init_data = torch.cat(ipc_data)
                print(init_data.size())
                # save into separate folders
                place_to_store = '{}/id{:03d}_batch{:03d}.pt'.format(args.init_path, args.ipc_start + ipc, start // 100)
                torch.save(init_data, place_to_store)
                del init_data, ipc_data

        for c in range(1000):
            os.remove(os.path.join(args.init_path, 'data_class{:03d}.pt'.format(c)))
    else: # init data
        for c in range(args.class_start, args.class_end):
            # init syn data
            init_data = []

            # path all
            if args.ipc_start != 0:
                path_all = torch.load(os.path.join(args.init_path, 'path_class{:03d}.pt'.format(c)))
            else:
                path_all = []

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            original_loader = class_loader(args, normalize, c)
            print(f'load original data of class {c} successfully')

            for idx, data in enumerate(original_loader):
                image = data[0].cuda()
                target = data[1].cuda() + c
                path = data[2]

                if len(init_data) >= args.init_num:
                    break

                image, path = filter(model_teacher, model_student, image, target, path, path_all, mode='simple')
                if image.size()[0] == 0:
                    continue

                if model_student is None:
                    init_data, path_all = update_data_path(init_data, path_all, image, path)
                    continue

                unrepeat, path_unrepeat = check_repitition(path, path_all)

                path = path_unrepeat
                image = image[unrepeat]
                target = target[unrepeat]
                if image.size()[0] == 0:
                    continue

                image, path = filter(model_teacher, model_student, image, target, path, path_all, mode='medium')

                if image.size()[0] > 0:
                    init_data, path_all = update_data_path(init_data, path_all, image, path)
            init_data = init_data[:args.init_num]
            path_all = path_all[:args.ipc_start + args.init_num]
            print(f'path_all len after medium data: {len(path_all)}')

            if len(init_data) < args.init_num and model_student is not None:
                print(f'class {c}\'s {args.init_num - len(init_data)} data init from hard original data.')
                original_loader_hard = class_loader(args, normalize, c)
                for idx, data in enumerate(original_loader_hard):
                    image_hard = data[0].cuda()
                    target_hard = data[1].cuda()
                    path_hard = data[2]

                    image_hard, path_hard = filter(model_teacher, model_student, image_hard, target_hard, path_hard, path_all, mode='hard')
                    
                    if image_hard.size()[0] == 0:
                        continue

                    unrepeat, path_unrepeat = check_repitition(path_hard, path_all)

                    path_hard = path_unrepeat
                    image_hard = image_hard[unrepeat]
                    if image_hard.size()[0] > 0:
                        init_data, path_all = update_data_path(init_data, path_all, image_hard, path_hard)
                    if len(init_data) >= args.init_num:
                        break
            print(f'path_all len after hard data: {len(path_all)}')

            if len(init_data) < args.init_num:
                print(f'class {c}\'s {args.init_num - len(init_data)} data init from other original data.')
                original_loader_other = class_loader(args, normalize, c)
                for idx, data in enumerate(original_loader_other):
                    image_other = data[0].cuda()
                    target_other = data[1].cuda()
                    path_other = data[2]

                    unrepeat, path_unrepeat = check_repitition(path_other, path_all)

                    path_other = path_unrepeat
                    image_other = image_other[unrepeat]
                    if image_other.size()[0] > 0:
                        init_data, path_all = update_data_path(init_data, path_all, image_other, path_other)
                    if len(init_data) >= args.init_num:
                        break
            print(f'path_all len after other data: {len(path_all)}')

            init_data = init_data[:args.init_num]
            path_all = path_all[:args.ipc_start + args.init_num]
            assert len(init_data) == args.init_num

            init_data = torch.cat(init_data)
            print(f'class {c} size: {init_data.size()}')
            init_data = denormalize(init_data)
            for cur_ipc in range(args.ipc_start, args.ipc_start + args.init_num):
                save_images(args, init_data[cur_ipc - args.ipc_start].unsqueeze(0), torch.tensor(c).unsqueeze(0).cuda(), cur_ipc)
            torch.save(init_data, os.path.join(args.init_path, 'data_class{:03d}.pt'.format(c)))
            torch.save(path_all, os.path.join(args.init_path, 'path_class{:03d}.pt'.format(c)))



def parse_args():
    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    """Data save flags"""
    parser.add_argument('--ipc-start', default=20, type=int)
    parser.add_argument('--init-num', default=30, type=int)
    parser.add_argument('--original-dir', type=str, default=None, help='path to load original data')
    parser.add_argument('--init-path', type=str, default=None, help='path to save init original data')
    parser.add_argument('--class-start', default=0, type=int)
    parser.add_argument('--class-end', default=1000, type=int)
    parser.add_argument('--form-batch', action='store_true', help='form batch')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18', help='arch name from pretrained torchvision models')
    parser.add_argument('--student-dir', type=str, default=None, help='path to pretrained student model dir')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    print('num to init = ', args.init_num)
    main_syn()
