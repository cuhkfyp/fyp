#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import math
from math import comb
import torch
from collections import OrderedDict
from numpy import linalg as LA
import copy

def maxFinder(w_locals, bc_arr, K): #w_locals => M, bc_arr => M
    w = np.array(w_locals)
    a = np.array(bc_arr)
    h = LA.norm(a, axis=1)

    idxs = h.argsort()[::-1][:K]
    ret_w = np.array(w[idxs])
    ret_h = np.array(h[idxs])

    return ret_w , ret_h  #ret_w => K, ret_h => K

def cFinder(h, M, K):  #h => K
    ret_arr=[]

    for i in range(K):
        c = math.log((1 + h[i] * h[i] * (M / K)), 2)
        ret_arr.append(c)

    return ret_arr  #ret_arr => K

def nFinder(C_arr, K, n):
    ret_arr = []
    lower = 0


    for j in range(K ):
        lowerMulti = 1
        for i in range(K ):
            if (i != j):
                lowerMulti *= C_arr[i]
        lower += lowerMulti

    for q in range(K ):
        upper = 1
        for i in range(K):
            if (i != q):
                upper *= C_arr[i]

        ret_arr.append(math.ceil((upper / lower) * n))


    return ret_arr


def qFinder(C_arr, N_arr, new_w_locals, K,glob):
    ret_arr=[]

    for index in range(K):
        #######get variable

        array_x = []
        flat_glob = []
        iwLen = np.array(new_w_locals[index]['layer_input.weight'].cpu().numpy()).shape
        ibLen = np.array(new_w_locals[index]['layer_input.bias'].cpu().numpy()).shape
        hwLen = np.array(new_w_locals[index]['layer_hidden.weight'].cpu().numpy()).shape
        hbLen = np.array(new_w_locals[index]['layer_hidden.bias'].cpu().numpy()).shape
        iw = np.prod(iwLen)
        ib = np.prod(ibLen)
        hw = np.prod(hwLen)
        hb = np.prod(hbLen)
        d = iw + ib + hw + hb
        target = 0

        #####perform binary search
        start = 0
        end = d

        while True:
            midpoint = (end + start) // 2
            check = math.log(comb(d, midpoint), 2) + 33 * midpoint
            if (end - start == 1):
                target = midpoint
                break
            elif check > N_arr[index] * C_arr[index]:
                end = midpoint
            else:
                start = midpoint
            # print(midpoint)



        for item in new_w_locals[index].keys():
            new_w_local_cpu=new_w_locals[index][item].cpu().numpy()
            glob_cpu = glob[item].cpu().numpy()
            k = new_w_local_cpu - glob_cpu
            flat_glob = np.concatenate((flat_glob,glob_cpu),axis=None)
            array_x = np.concatenate((array_x, k), axis=None)

        abs_x = np.array(np.abs(array_x))


        ind_that_set_to_zero = np.argpartition(abs_x, -target)[:-target]

        ind_that_quantize = np.argpartition(abs_x, -target)[-target:]

        array_x[ind_that_set_to_zero] = 0

        max_ele = max(array_x)
        min_ele = min(array_x)

        a=(max_ele-min_ele)/2**32

        array_x_temp = ((np.round((array_x - min_ele) / a)) * a) + min_ele

        array_x_temp[ind_that_set_to_zero] = 0

        if target == 0:
            array_x_temp = flat_glob
        elif target > 0:
            array_x_temp = np.add(flat_glob, array_x_temp)



        ret_arr.append(OrderedDict([
            ('layer_input.weight', torch.FloatTensor((np.reshape(array_x_temp[:iw], iwLen)).tolist())),
            ('layer_input.bias', torch.FloatTensor((np.reshape(array_x_temp[iw:iw + ib], ibLen)).tolist())),
            ('layer_hidden.weight', torch.FloatTensor((np.reshape(array_x_temp[iw + ib:iw + ib + hw], hwLen)).tolist())),
            ('layer_hidden.bias', torch.FloatTensor((np.reshape(array_x_temp[iw + ib + hw:], hbLen)).tolist()))
        ]))



    return ret_arr



