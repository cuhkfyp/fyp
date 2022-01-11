import numpy as np
import math
from numpy import linalg as LA
from math import comb
from collections import OrderedDict
import torch

def cFinder(bc_arr, K, M):  #bc_arr => M
    ret_arr = []
    h = LA.norm(np.array(bc_arr), axis=1)
    for i in range(M):
        c = math.log((1 + h[i] * h[i] * (M / K)), 2)
        ret_arr.append(c)
    return ret_arr  #ret_arr => M

def maxFinder(w_locals, K, C_arr):  #w_locals => M, C_arr => M
    w = []
    for i in range(len(w_locals)):
        temp = []
        for item in w_locals[i].keys():
            temp = (np.concatenate((temp, np.array(w_locals[i][item])), axis=None))
        w.append(temp)
    l2_arr = LA.norm(np.array(w), axis=1)

    idxs = l2_arr.argsort()[::-1][:K]
    ret_w = np.array(np.array(w_locals)[idxs])
    ret_l2 = np.array(l2_arr[idxs])
    ret_C = np.array(np.array(C_arr)[idxs])
    return (ret_w, ret_l2, ret_C)  #ret_w => K, ret_l2 => K, ret_C => K

def nFinder(l2_arr, C_arr, K, n):  #l2_arr => K, C_arr => K
    ret_arr = []
    lower = 0
    for j in range(K):
        lowerMulti = 1
        for i in range(K):
            if (i != j):
                lowerMulti *= C_arr[i]
        lower += lowerMulti * l2_arr[j]

    for q in range(K):
        upper = 1
        for i in range(K):
            if (i != q):
                upper *= C_arr[i]
        ret_arr.append( upper/lower * n * l2_arr[q])
    return ret_arr  #ret_arr => K

def qFinder_2nd(C_arr, N_arr, new_w_locals, K):
    ret_arr = []
    for index in range(K):
        #######get variable

        array_x = []
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

        # print("target is ",str(target))
        for item in new_w_locals[index].keys():
            array_x = np.concatenate((array_x, np.array(new_w_locals[index][item].cpu().numpy())), axis=None)

        abs_x = abs(array_x)
        ind = np.argpartition(abs_x, -target)[:-target]
        array_x[ind] = 0
        max_ele = max(array_x)
        min_ele = min(array_x)
        a = (max_ele - min_ele) / (2 ** 32)

        array_x_temp = ((np.round((array_x - min_ele) / a)) * a) + min_ele

        difference = np.sum(np.abs(array_x - array_x_temp))

        ret_arr.append(OrderedDict([
            ('layer_input.weight', torch.FloatTensor((np.reshape(array_x_temp[:iw], iwLen)).tolist())),
            ('layer_input.bias', torch.FloatTensor((np.reshape(array_x_temp[iw:iw + ib], ibLen)).tolist())),
            ('layer_hidden.weight', torch.FloatTensor((np.reshape(array_x_temp[iw + ib:iw + ib + hw], hwLen)).tolist())),
            ('layer_hidden.bias', torch.FloatTensor((np.reshape(array_x_temp[iw + ib + hw:], hbLen)).tolist()))
        ]))

    return ret_arr


def qFinder_1st(C_arr, new_w_locals, M, n):
    ret_arr = []
    for index in range(M):
        #######get variable

        array_x = []
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

            elif check > n * C_arr[index]:
                end = midpoint

            else:
                start = midpoint
            #print(midpoint)

        for item in new_w_locals[index].keys():
            array_x = np.concatenate((array_x, np.array(new_w_locals[index][item].cpu().numpy())), axis=None)
        #target = 1
        abs_x = np.array(np.abs(array_x))
        ind = np.argpartition(abs_x, -target)[:-target]
        array_x[ind] = 0
        max_ele = max(array_x)
        min_ele = min(array_x)
        a = (max_ele - min_ele) / (2 ** 32)

        array_x_temp = ((np.round((array_x - min_ele) / a)) * a) + min_ele

        difference = np.sum(np.abs(array_x - array_x_temp))

        ret_arr.append(OrderedDict([
            ('layer_input.weight', torch.FloatTensor((np.reshape(array_x_temp[:iw], iwLen)).tolist())),
            ('layer_input.bias', torch.FloatTensor((np.reshape(array_x_temp[iw:iw + ib], ibLen)).tolist())),
            ('layer_hidden.weight', torch.FloatTensor((np.reshape(array_x_temp[iw + ib:iw + ib + hw], hwLen)).tolist())),
            ('layer_hidden.bias', torch.FloatTensor((np.reshape(array_x_temp[iw + ib + hw:], hbLen)).tolist()))
        ]))

    return ret_arr
