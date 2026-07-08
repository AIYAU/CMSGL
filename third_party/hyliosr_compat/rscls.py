# -*- coding: utf-8 -*-
"""Minimal compatibility subset copied from the public HyLiOSR repository.

This file preserves the working CC-SGCL split / patch / evaluation behavior.
Only the subset required by standalone CC-SGCL is kept here.
"""

import copy
import numpy as np


class rscls:
    def __init__(self, im, gt, cls):
        if cls == 0:
            print('num of class not specified !!')
        self.im = copy.deepcopy(im)
        self.gt = copy.deepcopy(gt - 1)
        self.gt_b = copy.deepcopy(gt)
        self.cls = cls
        self.patch = 1
        self.imx, self.imy, self.imz = self.im.shape
        self.record = []
        self.sample = {}

    def padding(self, patch):
        self.patch = patch
        pad = self.patch // 2
        r1 = np.repeat([self.im[0, :, :]], pad, axis=0)
        r2 = np.repeat([self.im[-1, :, :]], pad, axis=0)
        self.im = np.concatenate((r1, self.im, r2))
        r1 = np.reshape(self.im[:, 0, :], [self.imx + 2 * pad, 1, self.imz])
        r2 = np.reshape(self.im[:, -1, :], [self.imx + 2 * pad, 1, self.imz])
        r1 = np.repeat(r1, pad, axis=1)
        r2 = np.repeat(r2, pad, axis=1)
        self.im = np.concatenate((r1, self.im, r2), axis=1)
        self.im = self.im.astype('float32')

    def locate_sample(self):
        sam = []
        for i in range(self.cls):
            _xy = np.array(np.where(self.gt == i)).T
            _sam = np.concatenate([_xy, i * np.ones([_xy.shape[0], 1])], axis=-1)
            try:
                sam = np.concatenate([sam, _sam], axis=0)
            except Exception:
                sam = _sam
        self.sample = sam.astype(int)

    def get_patch(self, xy):
        d = self.patch // 2
        x = xy[0]
        y = xy[1]
        try:
            self.im[x][y]
        except IndexError:
            return []
        x += d
        y += d
        sam = self.im[(x - d):(x + d + 1), (y - d):(y + d + 1)]
        return np.array(sam)

    def train_sample(self, pn):
        x_train, y_train = [], []
        self.locate_sample()
        _samp = self.sample
        for _cls in range(self.cls):
            _xy = _samp[_samp[:, 2] == _cls]
            np.random.shuffle(_xy)
            _xy = _xy[:pn, :]
            for xy in _xy:
                self.gt[xy[0], xy[1]] = 255
                x_train.append(self.get_patch(xy[:-1]))
                y_train.append(xy[-1])
        x_train, y_train = np.array(x_train), np.array(y_train)
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]
        return x_train, y_train.astype(int)

    def all_sample_row(self, sub=0):
        imx, imy = self.gt.shape
        fp = []
        for j in range(imy):
            xy = np.array([sub, j])
            fp.append(self.get_patch(xy))
        return np.array(fp)


def gtcfm(pre, gt, ncl):
    pre = np.uint8(pre)
    gt = np.uint8(gt)
    if gt.max() == 255:
        print('warning: max 255 !!')
    cf = np.zeros([ncl, ncl])
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j]:
                cf[pre[i, j] - 1, gt[i, j] - 1] += 1
    tmp1 = 0
    nsize = np.sum(gt != 0)
    for j in range(ncl):
        tmp1 = tmp1 + (cf[j, :].sum() / nsize) * (cf[:, j].sum() / nsize)
    cfm = np.zeros((ncl + 2, ncl + 1))
    cfm[:-2, :-1] = cf
    oa = 0
    for i in range(ncl):
        if cf[i, :].sum():
            cfm[i, ncl] = cf[i, i] / cf[i, :].sum()
        if cf[:, i].sum():
            cfm[ncl, i] = cf[i, i] / cf[:, i].sum()
        oa += cf[i, i]
    cfm[-1, 0] = oa / nsize
    cfm[-1, 1] = (cfm[-1, 0] - tmp1) / (1 - tmp1)
    cfm[-1, 2] = cfm[ncl, :-1].mean()
    cfm[-1, 3] = cfm[:-2, ncl].mean()
    oa = cfm[-1, 0]
    aa = cfm[-1, 2]
    kappa = cfm[-1, 1]
    print('oa: ', format(oa, '.5'), ' kappa: ', format(kappa, '.5'),
          ' aa/pa: ', format(aa, '.5'), ' ua: ', format(cfm[-1, 3], '.5'))
    print('AA is :')
    osr = 0
    for acc in cfm[ncl, :-1]:
        print(acc)
        osr = acc
    return cfm, oa, aa, kappa, osr


def make_sample(sample, label):
    a = np.flip(sample, 1)
    b = np.flip(sample, 2)
    c = np.flip(b, 1)
    newsample = np.concatenate((a, b, c, sample), axis=0)
    newlabel = np.concatenate((label, label, label, label), axis=0)
    return newsample, newlabel
