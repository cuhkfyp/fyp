#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
import utils.BC as BC
import utils.BN2 as BN2
import utils.BC_BN2 as BC_BN2
import utils.BN2_C as BN2_C
import numpy as np
from collections import OrderedDict
import math
from math import comb

from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img



if __name__ == '__main__':
    # parse args
    args = args_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
#    print(dict_users[0])
#    print("###################################################################################################################")
#    print(dict_users[1])
#    exit(0)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).to(args.device)

    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    loss_test_array=[]
    acc_test_array=[]
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    test_mode = "BC_BN2"  # for test only / mode of scheduling scheme
    Kc =10
    K = int(args.frac * args.num_users)  # for test only / number of selected user

    M = args.num_users
    n = 5000  # for test only / total transmission time in that iter

    #print("this is the case that  total-local")
    #print("and without the zero line")
    #print("this is the case that only total")
    print(test_mode)
    print(args.num_users*args.frac)
    print("kc is",end=" ")
    print(Kc)




    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]


    for iter in range(args.epochs):
        loss_locals = []

        if not args.all_clients:
            w_locals = []
        m = max(int(args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))

###################################################################################################################
        bc_arr = np.random.randn(len(w_locals), 2) / np.sqrt(2)

        n_new_w_locals = []
        if(test_mode == "BC"):
            #bc_arr = np.random.randn(len(w_locals),2)/np.sqrt(2)
            (new_w_locals, h) = BC.maxFinder(w_locals, bc_arr, K)
            C_arr = BC.cFinder(h, M, K)
            N_arr = BC.nFinder(C_arr, K, n)
            n_new_w_locals = BC.qFinder(C_arr, N_arr, new_w_locals, K,w_glob)  #then should be w_glob = FedAvg(n_new_w_locals)

        if(test_mode == "BN2"): # BN2
            if iter>=64:
                hi=0
            (new_w_locals, l2_arr,bc_arr_2) = BN2.maxFinder(w_locals, K,w_glob,bc_arr)
            #bc_arr = np.random.randn(K, 2) / np.sqrt(2)
            C_arr = BN2.cFinder(bc_arr_2, K, M)
            N_arr = BN2.nFinder(l2_arr, C_arr, K, n)
            n_new_w_locals = BN2.qFinder(C_arr, N_arr, new_w_locals, K,w_glob)

        if(test_mode == "BC_BN2"): # BC-BN2
            #bc_arr = np.random.rand(M, 2) / np.sqrt(2)
            (new_w_locals, l2_arr, h) = BC_BN2.maxFinder(w_locals, bc_arr, Kc, K,w_glob)
            C_arr = BC_BN2.cFinder(h, K, M)
            N_arr = BC_BN2.nFinder(l2_arr, C_arr, K, n)
            n_new_w_locals = BC_BN2.qFinder(C_arr, N_arr, new_w_locals, K,w_glob)

        if (test_mode == "BN2_C"):  # BN2-C
            #bc_arr = np.random.randn(M, 2) / np.sqrt(2)
            C_arr = BN2_C.cFinder(bc_arr, K, M)
            w_1st = BN2_C.qFinder_1st(C_arr, w_locals, M, n, w_glob)
            (new_w_locals, l2_arr, new_C_arr) = BN2_C.maxFinder(w_1st, K, C_arr, w_locals)
            N_arr = BN2_C.nFinder(l2_arr, new_C_arr, K, n)
            n_new_w_locals = BN2_C.qFinder_2nd(new_C_arr, N_arr, new_w_locals, K, w_glob)

####################################################################################################################
        # update global weights locals-glob=detla
        #w_glob = FedAvg(w_locals)



        w_glob = FedAvg(n_new_w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        #loss_avg = sum(loss_locals) / len(loss_locals)
        #print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        #loss_train.append(loss_avg)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        loss_test_array.append(loss_test)
        acc_test_array.append(acc_test)
        print('Round {:3d}, global loss {:.3f}, global acc {:.3f}'.format(iter, loss_test, acc_test))


    # plot loss curve
    plt.figure()
    plt.ylim([0, 5])
    plt.plot(range(len(loss_test_array)), loss_test_array,color=(255/255,105/255,180/255))
    plt.ylabel('test_loss')
    plt.savefig('./save/___fed__{}_{}___{}_{}__C{}_iid{}_local_ep_{}_local_batch_{}_lr_{}_loss_fig_.png'.format(args.dataset, test_mode, args.model, args.epochs, args.frac, args.iid,args.local_ep,args.local_bs,args.lr))

    plt.figure()
    plt.ylim([0, 100])
    plt.plot(range(len(acc_test_array)), acc_test_array, color=(255/255,105/255,180/255))
    plt.ylabel('test_acc')
    plt.savefig('./save/___fed__{}_{}_{}___{}__C{}_iid{}_local_ep_{}_local_batch_{}_lr_{}_acc_fig_.png'.format(args.dataset, test_mode, args.model, args.epochs, args.frac, args.iid,args.local_ep,args.local_bs,args.lr))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    other_data = "./store/outbcarr256/___train_acc_test_acc_test_acc_array_test_loss_array_{}_{}_{}_ep{}_frac{}_iid{}_local_ep{}_local_bs{}_lr{}_kc_{}".format(args.dataset, test_mode, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.lr,Kc)
    #other_data = "./store/nsmall5000/___train_acc_test_acc_test_acc_array_test_loss_array_{}_{}_{}_ep{}_frac{}_iid{}_local_ep{}_local_bs{}_lr{}_kc_{}".format(args.dataset, test_mode, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.lr,Kc)
    np.savez(other_data, Training_accuracy=acc_train, Testing_acc=acc_test, train_loss=loss_train,testing_acc_array=acc_test_array, testing_loss_array=loss_test_array)
