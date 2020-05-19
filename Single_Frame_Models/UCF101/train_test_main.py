# Author: Paritosh

import torch
from torch.utils.data import DataLoader
from dataloader_ucf101_centralframe import VideoDataset_2
from models.model_2 import VGG_Backbone
from models.model_2 import Cls_branch
from models.model_2 import Sidetask_branch_resnext
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from opts import *
import os
from sklearn.metrics import accuracy_score
from models.ResNeXt3D.opts_resnext import parse_opts
from models.ResNeXt3D.resnext_model_generator import generate_model
from collections import OrderedDict

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

with_hallu_task = True
with_cls = True

hallu_task_wt = 50

Sigmoid = nn.Sigmoid()

def trainphase(train_dataloader, optimizer, criterions, epoch):

    criterion_cls = criterions['criterion_cls']; criterion_reg = criterions['criterion_reg']

    model_2dcnn.train()
    if with_cls:
        model_linear_layers.train()
    if with_hallu_task:
        model_3dcnn.eval()
        model_fc6_feats.train()
    iteration = 0
    for data in train_dataloader:
        image = data['2dcnn_image'].cuda()

        true_labels = data['label'].cuda()

        pool5_feats = model_2dcnn(image)

        if with_hallu_task:
            clips = data['video'].transpose_(1, 2).cuda()

            with torch.no_grad():
                true_3d_feats = model_3dcnn(clips)
            hallucinated_3d_feats = model_fc6_feats(pool5_feats)

        if with_cls: # adjust like you wish (want to concatenate hallucinated features or not)
            # if with_hallu_task:
            #     concat_feats = torch.cat((pool5_feats, hallucinated_3d_feats), 1)
            # else:
            concat_feats = pool5_feats
            pred_labels = model_linear_layers(concat_feats)

        optimizer.zero_grad()
        loss = 0
        if with_cls:
            loss_cls = criterion_cls(pred_labels, true_labels)
            loss = loss + loss_cls
        if with_hallu_task:
            loss_reg = hallu_task_wt*criterion_reg(Sigmoid(hallucinated_3d_feats), Sigmoid(true_3d_feats))
            loss = loss + loss_reg
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print('Epoch: ', epoch, ' Iter: ', iteration, '     Loss: ', loss.data.cpu().numpy(), end="")
            if with_hallu_task and with_cls:
                print('     Cls Loss: ', loss_cls.data.cpu().numpy(), end="")
                print('     Hallu_task Loss: ', loss_reg.data.cpu().numpy(), end="")
            print(' ')

        if iteration == 100:
            break
        iteration += 1


def test_phase(test_dataloader, criterions):
    criterion_cls = criterions['criterion_cls']; criterion_reg = criterions['criterion_reg']
    test_loss = 0
    hallu_loss = 0
    with torch.no_grad():
        pred_labels = []; true_labels = []

        model_2dcnn.eval()
        if with_cls:
            model_linear_layers.eval()
        if with_hallu_task:
            model_3dcnn.eval()
            model_fc6_feats.eval()
        softmax_layer = nn.Softmax(dim=1)
        iteration = 0
        for data in test_dataloader:
            image = data['2dcnn_image'].cuda()

            true_labels.extend(data['label'].numpy())
            temp_true_labels = data['label'].cuda()
            pool5_feats = model_2dcnn(image)

            if with_hallu_task:
                clips = data['video'].transpose_(1, 2).cuda()

                true_3d_feats = model_3dcnn(clips)
                hallucinated_3d_feats = model_fc6_feats(pool5_feats)
                test_loss = test_loss + hallu_task_wt*criterion_reg(Sigmoid(true_3d_feats), Sigmoid(hallucinated_3d_feats))
                hallu_loss = hallu_loss + hallu_task_wt * criterion_reg(Sigmoid(true_3d_feats), Sigmoid(hallucinated_3d_feats))

            if with_cls:# adjust like you wish (want to concatenate hallucinated features or not)
                # concat_feats = torch.cat((pool5_feats, hallucinated_3d_feats), 1)
                # if with_hallu_task:
                #     concat_feats = torch.cat((pool5_feats, hallucinated_3d_feats), 1)
                # else:
                concat_feats = pool5_feats
                temp_pred_labels = model_linear_layers(concat_feats)
                test_loss = test_loss + criterion_cls(temp_pred_labels, temp_true_labels)
                temp_pred_labels = softmax_layer(temp_pred_labels).data.cpu().numpy()
                for i in range(len(temp_pred_labels)):
                    pred_labels.extend(np.argwhere(temp_pred_labels[i] == max(temp_pred_labels[i]))[0])

            iteration += 1

        if with_cls:
            accuracy = accuracy_score(pred_labels, true_labels)
            print('Test Accuracy: ', accuracy)

        if with_hallu_task:
            print('Test-time Hallucination Loss: ', hallu_loss)

    return test_loss


def save_model(m, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(m.state_dict(), model_path)


def main():
    learning_rate = 0.0001
    lr_multiplier_fc6 = 1
    if with_hallu_task and with_cls:
        print('Optimizing: side task + cls task')
        optimizer = optim.Adam([{'params': model_2dcnn.parameters()},
                               {'params': model_linear_layers.parameters()},
                               {'params': model_fc6_feats.parameters(), 'lr': learning_rate * lr_multiplier_fc6}],
                              lr=learning_rate)

    elif with_cls:
        print('Optimizing: cls task')
        optimizer = optim.Adam([{'params': model_2dcnn.parameters()},
                               {'params': model_linear_layers.parameters()}],
                              lr=learning_rate)
    else:
        print('Optimizing: side task')
        optimizer = optim.Adam([{'params': model_2dcnn.parameters()},
                               {'params': model_fc6_feats.parameters(), 'lr': learning_rate * lr_multiplier_fc6}],
                              lr=learning_rate)

    criterions = {}
    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_reg = nn.MSELoss().cuda()
    criterions['criterion_cls'] = criterion_cls
    criterions['criterion_reg'] = criterion_reg

    train_dataset = VideoDataset_2('train'); test_dataset = VideoDataset_2('test')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    print('Length of train dataloader: ', len(train_dataloader))
    print('Length of test_dataloader: ', len(test_dataloader))

    for epoch in np.arange(0,50):
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])
        trainphase(train_dataloader, optimizer, criterions, epoch)

        # save models
        # if (epoch+1) % 50 == 0:
        #     storing_dir = '../../trained_models/hmdb_51/wst/'
        #     save_model(model_2dcnn, 'vggbn', epoch, storing_dir)
        #     if with_hallu_task:
        #         save_model(model_fc6_feats, 'hallu_task_fc6', epoch, storing_dir)
        #     if with_cls:
        #         save_model(model_linear_layers, 'cls_fc6_fc7', epoch, storing_dir)

        test_loss = test_phase(test_dataloader, criterions)

        print('Test_loss: ', test_loss)

        if (epoch+1) % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/10


if __name__ == '__main__':
    model_2dcnn = VGG_Backbone()
    model_2dcnn_pretrained_dict = torch.load('/home/paritosh/pytorch_tutos/aar/models/VGG/vgg11_bn.pth')
    model_2dcnn_dict = model_2dcnn.state_dict()
    model_2dcnn_pretrained_dict = {k: v for k, v in model_2dcnn_pretrained_dict.items() if k in model_2dcnn_dict}
    model_2dcnn_dict.update(model_2dcnn_pretrained_dict)
    model_2dcnn.load_state_dict(model_2dcnn_dict)
    model_2dcnn = model_2dcnn.cuda()

    if with_cls:
        model_linear_layers = Cls_branch()
        model_linear_layers = model_linear_layers.cuda()

    if with_hallu_task:
        ################################## loading Resnet 3D ###################################
        opt = parse_opts()
        # opt.mean = get_mean()
        opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        opt.sample_size = 112
        opt.sample_duration = 16
        opt.n_classes = 101

        model_3dcnn = generate_model(opt)
        print('loading model {}'.format(opt.model))
        model_data = torch.load(opt.model)
        assert opt.arch == model_data['arch']
        w = torch.load(opt.model)
        # print(w)
        ############

        state_dict = w['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model_3dcnn.load_state_dict(new_state_dict)
        model_3dcnn = model_3dcnn.cuda()
        ########################################################################################

        model_fc6_feats = Sidetask_branch_resnext()#Sidetask_branch()#
        model_fc6_feats = model_fc6_feats.cuda()

    main()