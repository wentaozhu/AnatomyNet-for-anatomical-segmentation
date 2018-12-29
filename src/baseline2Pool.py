
# coding: utf-8

# In[1]:


# import matplotlib
# matplotlib.use('Agg')
# get_ipython().magic(u'matplotlib inline')
# import matplotlib.pyplot as plt
# plt.rcParams['image.cmap'] = 'gray' 
from glob import glob
import SimpleITK as sitk
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# from ipywidgets import interact, interactive
# from ipywidgets import widgets
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import os
import numpy as np
from tqdm import tqdm

import torch as t
# t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
from torch.utils import data
from torchvision import transforms as tsf

TRAIN_PATH = './data/trainpddca15_crp_v2_pool2.pth'
TEST_PATH = './data/testpddca15_crp_v2_pool2.pth'
CET_PATH = './data/trainpddca15_cet_crp_v2_pool2.pth'
PET_PATH = './data/trainpddca15_pet_crp_v2_pool2.pth'
os.environ["CUDA_VISIBLE_DEVICES"]="6"


# In[2]:


import SimpleITK as sitk
import math
from scipy.ndimage.interpolation import zoom
def getdatamaskfilenames(path, maskname):
    data, masks_data = [], []
    for pth in path: # get data files and mask files
        maskfiles = []
        for seg in maskname:
            if os.path.exists(os.path.join(pth, './structures/'+seg+'_crp_v2.npy')):
                maskfiles.append(os.path.join(pth, './structures/'+seg+'_crp_v2.npy'))
            else:
                print('missing annotation', seg, pth.split('/')[-1])
                maskfiles.append(None)
        data.append(os.path.join(pth, 'img_crp_v2.npy'))
        masks_data.append(maskfiles)
    return data, masks_data
def imfit(img, newz, newy, newx):
    z, y, x = img.shape
    retimg = np.zeros((newz, newy, newx), img.dtype)
    bz, ez = newz/2, newz/2+1
    while ez - bz < z:
        if bz - 1 >= 0:
            bz -= 1
        if ez - bz < z:
            if ez + 1 <= z:
                ez += 1
    by, ey = newy/2, newy/2+1
    while ey - by < y:
        if by - 1 >= 0:
            by -= 1
        if ey - by < y:
            if ey + 1 <= y:
                ey += 1
    bx, ex = newx/2, newx/2+1
    while ex - bx < x:
        if bx - 1 >= 0:
            bx -= 1
        if ex - bx < x:
            if ex + 1 <= x:
                ex += 1
    retimg[bz:ez, by:ey, bx:ex] = img
    return retimg
def getdatamask(data, mask_data, debug=False): # read data and mask, reshape
    datas = []
    for fnm, masks in tqdm(zip(data, mask_data)):
        item = {}
        img = np.load(fnm) # z y x
        nz, ny, nx = img.shape
#         if nz > 300 or ny > 300 or nx > 300: 
#             print(fnm, nx, ny, nz)
#             assert 1==0
        tnz, tny, tnx = math.ceil(nz/8.)*8., math.ceil(ny/8.)*8., math.ceil(nx/8.)*8.
        img = imfit(img, int(tnz), int(tny), int(tnx)) #zoom(img, (tnz/nz,tny/ny,tnx/nx), order=2, mode='nearest')
        item['img'] = t.from_numpy(img)
        item['mask'] = []
        for idx, maskfnm in enumerate(masks):
            if maskfnm is None: 
                ms = np.zeros((nz, ny, nx), np.uint8)
            else: 
                ms = np.load(maskfnm).astype(np.uint8)
                assert ms.min() == 0 and ms.max() == 1
            mask = imfit(ms, int(tnz), int(tny), int(tnx)) #zoom(ms, (tnz/nz,tny/ny,tnx/nx), order=0, mode='constant')
            item['mask'].append(mask)
        assert len(item['mask']) == 9
        item['name'] = str(fnm)#.split('/')[-1]
        datas.append(item)
    return datas
def process(path='/data/wtzhu/dataset/pddca18/', debug=False):
    trfnmlst, trfnmlstopt, tefnmlstoff, tefnmlst = [], [], [], [] # get train and test files
    train_files, train_filesopt, test_filesoff, test_files = [], [], [], [] # MICCAI15 and MICCAI16 use different test
    for pid in os.listdir(path):
        if '0522c0001' <= pid <= '0522c0328':
            trfnmlst.append(pid)
            train_files.append(os.path.join(path, pid))
        elif '0522c0329' <= pid <= '0522c0479':
            trfnmlstopt.append(pid)
            train_filesopt.append(os.path.join(path, pid))
        elif '0522c0555' <= pid <= '0522c0746':
            tefnmlstoff.append(pid)
            test_filesoff.append(os.path.join(path, pid))
        elif '0522c0788' <= pid <= '0522c0878':
            tefnmlst.append(pid)
            test_files.append(os.path.join(path, pid))
        else:
            print(pid)
            assert 1 == 0
    print('train file names', trfnmlst)
    print('optional train file names', trfnmlstopt)
    print('offsite test file names', tefnmlstoff)
    print('onsite test file names', tefnmlst)
    print('Total train files', len(train_files), 'total test files', len(test_files))
    print('Train optional files', len(train_filesopt), 'test optional files', len(test_filesoff))
    assert len(trfnmlst) == 25 and len(trfnmlstopt) == 8 and len(tefnmlstoff) == 10 and len(tefnmlst) == 5
    assert len(train_files) == 25 and len(train_filesopt) == 8 and len(test_filesoff) == 10 and     len(test_files) == 5
    structurefnmlst = ('BrainStem', 'Chiasm', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R',                        'Submandibular_L', 'Submandibular_R')
    train_data, train_masks_data = getdatamaskfilenames(train_files, structurefnmlst)
    train_dataopt, train_masks_dataopt = getdatamaskfilenames(train_filesopt, structurefnmlst)
    test_data, test_masks_data = getdatamaskfilenames(test_files, structurefnmlst)
    test_dataoff, test_masks_dataoff = getdatamaskfilenames(test_filesoff, structurefnmlst)
    return getdatamask(train_data+train_dataopt+test_data,                        train_masks_data+train_masks_dataopt+test_masks_data,debug=debug),            getdatamask(test_dataoff, test_masks_dataoff,debug=debug)
def processCET(path='/data/wtzhu/dataset/HNCetuximabclean/', debug=False):
    trfnmlst = [] # get train and test files
    train_files = [] # MICCAI15 and MICCAI16 use different test
    for pid in os.listdir(path):
        trfnmlst.append(pid)
        train_files.append(os.path.join(path, pid))
    print('train file names', trfnmlst)
    print('Total train files', len(train_files))
    structurefnmlst = ('BrainStem', 'Chiasm', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R',                        'Submandibular_L', 'Submandibular_R')
    train_data, train_masks_data = getdatamaskfilenames(train_files, structurefnmlst)
    return getdatamask(train_data, train_masks_data,debug=debug)
# You can skip this if you have alreadly done it.
if not os.path.isfile(TRAIN_PATH):
    train_data, test_data = process('/data/wtzhu/dataset/pddca18/')
    print('use train', len(train_data), 'use test', len(test_data))
    t.save(train_data, TRAIN_PATH)
    t.save(test_data, TEST_PATH)
if not os.path.isfile(CET_PATH):
    train_data, test_data = process('/data/wtzhu/dataset/pddca18/')
    print('use train', len(train_data), 'use test', len(test_data))
    data = processCET('/data/wtzhu/dataset/HNCetuximabclean/')
    print('use ', len(data))
    t.save(data+train_data, CET_PATH)
if not os.path.isfile(PET_PATH):
    train_data, test_data = process('/data/wtzhu/dataset/pddca18/')
    print('use train', len(train_data), 'use test', len(test_data))
    data = processCET('/data/wtzhu/dataset/HNCetuximabclean/')
    print('use ', len(data))
    petdata = processCET('/data/wtzhu/dataset/HNPETCTclean/')
    print('use ', len(petdata))
    t.save(data+train_data+petdata, PET_PATH)


# In[3]:


class DatasetStg1():
    def __init__(self,path, istranform=True, alpha=1000, sigma=30, alpha_affine=0.04, istest=False):
        self.datas = t.load(path)
        self.ist = istranform
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.istest = istest
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy().astype(np.float32)
        if not self.istest:
            for mask in data['mask']: # for multi-task 
                if mask is None: 
                    print(data['name'])
                    assert 1 == 0
        if not self.ist: #[::2, ::2, ::2]
            masklst = []
            for mask in data['mask']:
                if mask is None: mask = np.zeros((1,img.shape[0],img.shape[1],img.shape[2])).astype(np.uint8)
                masklst.append(mask.astype(np.uint8).reshape((1,img.shape[0],img.shape[1],img.shape[2]))) 
            mask0 = np.zeros_like(masklst[0]).astype(np.uint8)
            for mask in masklst:
                mask0 = np.logical_or(mask0, mask).astype(np.uint8)
            mask0 = 1 - mask0
            return t.from_numpy(img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))),                    t.from_numpy(np.concatenate([mask0]+masklst, axis=0)), True
        im_merge = np.concatenate([img[...,None]]+[mask.astype(np.float32)[...,None] for mask in data['mask']],                                  axis=3)
        # Apply transformation on image
        im_merge_t, new_img = self.elastic_transform3Dv2(im_merge,self.alpha,self.sigma,min(im_merge.shape[1:-1])*self.alpha_affine)
        # Split image and mask ::2, ::2, ::2
        im_t = im_merge_t[...,0]
        im_mask_t = im_merge_t[..., 1:].astype(np.uint8).transpose(3, 0, 1, 2)
        mask0 = np.zeros_like(im_mask_t[0, :, :, :]).reshape((1,)+im_mask_t.shape[1:]).astype(np.uint8)
        im_mask_t_lst = []
        flagvect = np.ones((10,), np.float32)
        retflag = True
        for i in range(9):
            im_mask_t_lst.append(im_mask_t[i,:,:,:].reshape((1,)+im_mask_t.shape[1:]))
            if im_mask_t[i,:,:,:].max() != 1: 
                retflag = False
                flagvect[i+1] = 0
            mask0 = np.logical_or(mask0, im_mask_t[i,:,:,:]).astype(np.uint8)
        if not retflag: flagvect[0] = 0
        mask0 = 1 - mask0
        return t.from_numpy(im_t.reshape((1,)+im_t.shape[:3])),                t.from_numpy(np.concatenate([mask0]+im_mask_t_lst, axis=0)), flagvect
    def __len__(self):
        return len(self.datas)
    def elastic_transform3Dv2(self, image, alpha, sigma, alpha_affine, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         From https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
        """
        # affine and deformation must be slice by slice and fixed for slices
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape # image is contatenated, the first channel [:,:,:,0] is the image, the second channel 
        # [:,:,:,1] is the mask. The two channel are under the same tranformation.
        shape_size = shape[:-1] # z y x
        # Random affine
        shape_size_aff = shape[1:-1] # y x
        center_square = np.float32(shape_size_aff) // 2
        square_size = min(shape_size_aff) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size],                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        new_img = np.zeros_like(image)
        for i in range(shape[0]):
            new_img[i,:,:,0] = cv2.warpAffine(image[i,:,:,0], M, shape_size_aff[::-1],                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
            for j in range(1, 10):
                new_img[i,:,:,j] = cv2.warpAffine(image[i,:,:,j], M, shape_size_aff[::-1], flags=cv2.INTER_NEAREST,                                                  borderMode=cv2.BORDER_TRANSPARENT, borderValue=0)
        dx = gaussian_filter((random_state.rand(*shape[1:-1]) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape[1:-1]) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape_size_aff[1]), np.arange(shape_size_aff[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        new_img2 = np.zeros_like(image)
        for i in range(shape[0]):
            new_img2[i,:,:,0] = map_coordinates(new_img[i,:,:,0], indices, order=1, mode='constant').reshape(shape[1:-1])
            for j in range(1, 10):
                new_img2[i,:,:,j] = map_coordinates(new_img[i,:,:,j], indices, order=0, mode='constant').reshape(shape[1:-1])
        return np.array(new_img2), new_img
traindataset = DatasetStg1(PET_PATH, istranform=True)
traindataloader = t.utils.data.DataLoader(traindataset,num_workers=10,batch_size=1, shuffle=True)
testdataset = DatasetStg1(TEST_PATH, istranform=False)
testdataloader = t.utils.data.DataLoader(testdataset,num_workers=10,batch_size=1)
print(len(traindataloader), len(testdataloader))


# In[4]:


# sub-parts of the U-Net model
from torch import nn
import torch.nn.functional as F
from scipy.spatial.distance import dice
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, in_ch2=None, out_ch2=None, stride=2, bias=False):
        super(double_conv, self).__init__()
        if in_ch2 is None: in_ch2, out_ch2 = in_ch, out_ch
        if bias:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=stride, bias=True),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_ch2, out_ch2, 3, padding=1, bias=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch2, out_ch2, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        x = self.conv(x)
        return x
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x
class down(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
#             nn.MaxPool3d(2),
            double_conv(in_ch, out_ch, stride=stride)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch, in_ch2=None, out_ch2=None, in_ch3=None, out_ch3=None,                  bilinear=False, bias=False, stride=2):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if in_ch2 == None:
            in_ch2, out_ch2 = in_ch, out_ch
        if in_ch3 == None:
            in_ch3, out_ch3 = in_ch2, out_ch2
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
        elif stride==2:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=stride)
        self.stride = stride
        self.conv = double_conv(in_ch2, out_ch2, in_ch3, out_ch3, stride=1, bias=bias)
    def forward(self, x1, x2):
        if self.stride == 2:
            x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = t.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
        self.down1 = down(n_channels, 48, 2) # 1/2
        self.down2 = down(48, 96, 2) # 1/4
        self.down3 = down(96, 112, 1) # 1/4
        self.down4 = down(112, 128, 1)
        self.up1 = up(240, 112, stride=1) 
        self.up2 = up(208, 96, stride=1) # 1/4
        self.up3 = up(96, 48) # 1/2
        self.up4 = up(48, 24, 25, 24, 24, n_classes, bias=True) # 1
#         self.outc = outconv(12, n_classes)
    def forward(self, x):
#         x1 = self.inc(x)
#         print(x.size())
        x2 = self.down1(x) # 1/2
#         print(x2.size())
        x3 = self.down2(x2) # 1/4
#         print(x3.size())
        x4 = self.down3(x3) # 1/8
#         print(x4.size())
        x5 = self.down4(x4) # 1/8
#         print(x5.size())
        outx = self.up1(x5, x4) # 1/8
#         print(x5.size(), x4.size(), outx.size())
        outx = self.up2(outx, x3) # 1/4
#         print(x3.size(), outx.size())
        outx = self.up3(outx, x2) # 1/2
#         print(x2.size(), outx.size())
        outx = self.up4(outx, x) # 1
#         print(x.size(), outx.size())
#         outx = self.outc(outx)
#         print(outx.size())
        outx = F.softmax(outx, dim=1)
#         print(outx.size())
        return outx
# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss_wmask(y_pred, y_true, flagvec):
    alpha = 0.5
    beta  = 0.5
    ones = t.ones_like(y_pred) #K.ones(K.shape(y_true))
#     print(type(ones.data), type(y_true.data), type(y_pred.data), ones.size(), y_pred.size())
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true.type(t.cuda.FloatTensor)
    g1 = ones-g0
    num = t.sum(t.sum(t.sum(t.sum(p0*g0, 4),3),2),0) #(0,2,3,4)) #K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*t.sum(t.sum(t.sum(t.sum(p0*g1,4),3),2),0) + beta*t.sum(t.sum(t.sum(t.sum(p1*g0,4),3),2),0) #(0,2,3,4))

    T = t.sum((num * flagvec.cuda())/(den+1e-5))

#     Ncl = y_pred.size(1)*1.0
#     print(Ncl, T)
    return t.sum(flagvec.cuda())-T
def caldice(y_pred, y_true):
#     print(y_pred.sum(), y_true.sum())
    y_pred = y_pred.data.cpu().numpy().transpose(1,0,2,3,4) # inference should be arg max
    y_pred = np.argmax(y_pred, axis=0).squeeze() # z y x
    y_true = y_true.data.numpy().transpose(1,0,2,3,4).squeeze() # .cpu()
    avgdice = []
    y_pred_1 = y_pred==1
    y_true_1 = y_true[1,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==2
    y_true_1 = y_true[2,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==3
    y_true_1 = y_true[3,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==4
    y_true_1 = y_true[4,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==5
    y_true_1 = y_true[5,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==6
    y_true_1 = y_true[6,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==7
    y_true_1 = y_true[7,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==8
    y_true_1 = y_true[8,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==9
    y_true_1 = y_true[9,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    for dice in avgdice: 
        if dice != -1:
            assert 0 <= dice <= 1
    return avgdice
model = UNet(1,9+1).cuda()
lossweight = np.array([2.22, 1.31, 1.99, 1.13, 1.93, 1.93, 1.0, 1.0, 1.90, 1.98], np.float32)
savename = './model/unet10pool2e2e_pet_wmask_rmsp_'


# In[5]:


optimizer = t.optim.RMSprop(model.parameters(),lr = 2e-3)
maxloss = [0 for _ in range(9)]
for epoch in range(150):
    tq = tqdm(traindataloader, desc='loss', leave=True)
    trainloss = 0
    for x_train, y_train, flagvec in tq:
#         print(x_train.size(), y_train.size(), flagvec.size())
        x_train = t.autograd.Variable(x_train.cuda())
        y_train = t.autograd.Variable(y_train.cuda())
        optimizer.zero_grad()
        o = model(x_train)
        loss = tversky_loss_wmask(o, y_train, flagvec*t.from_numpy(lossweight))
        loss.backward()
        optimizer.step()
        tq.set_description("epoch %i loss %f" % (epoch, loss.item()))
        tq.refresh() # to show immediately the update
        trainloss += loss.item()
        del loss, x_train, y_train, o
    testtq = tqdm(testdataloader, desc='test loss', leave=True)
    testloss = [0 for _ in range(9)]
    for x_test, y_test, _ in testtq:
#         print(x_test.numpy().shape)
        with t.no_grad():
            x_test = t.autograd.Variable(x_test.cuda())
#             y_test = t.autograd.Variable(y_test.cuda())
        o = model(x_test)
        loss = caldice(o, y_test)
        testtq.set_description("epoch %i test loss %f" % (epoch, sum(loss)/9))
        testtq.refresh() # to show immediately the update
        testloss = [l+tl for l,tl in zip(loss, testloss)]
        del x_test, y_test, o
    testloss = [l / len(testtq) for l in testloss]
    for cls in range(9):
        if maxloss[cls] < testloss[cls]:
            maxloss[cls] = testloss[cls]
            t.save(model, savename+str(cls+1))
    print('epoch %i TRAIN loss %.4f' % (epoch, trainloss/len(tq)))
    print('test loss %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % tuple(testloss))
    print('best test loss %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % tuple(maxloss))
    if epoch % 10 == 0:
        testloss = [0 for _ in range(9)]
        ntest = [0 for _ in range(9)]
        testtq = tqdm(traindataloader, desc='loss', leave=True)
        for x_test, y_test, _ in testtq:
    #         print(x_test.numpy().shape)
            with t.no_grad():
                x_test = t.autograd.Variable(x_test.cuda())
#                 y_test = t.autograd.Variable(y_test.cuda())
            o = model(x_test)
            loss = caldice(o, y_test)
            testtq.set_description("epoch %i test loss %f" % (epoch, sum(loss)/9))
            testtq.refresh() # to show immediately the update
            testloss = [l+tl if l != -1 else tl for l,tl in zip(loss, testloss)]
            ntest = [n+1 if l != -1 else n for l, n in zip(loss, ntest)]
            del x_test, y_test, o
        testloss = [l / n for l,n in zip(testloss, ntest)]
        print('train loss %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % tuple(testloss))


# In[6]:


optimizer = t.optim.SGD(model.parameters(), 1e-3, momentum = 0.9)#, weight_decay = 1e-4)
for epoch in range(50):
    tq = tqdm(traindataloader, desc='loss', leave=True)
    trainloss = 0
    for x_train, y_train, flagvec in tq:
        x_train = t.autograd.Variable(x_train.cuda())
        y_train = t.autograd.Variable(y_train.cuda())
        optimizer.zero_grad()
        o = model(x_train)
        loss = tversky_loss_wmask(o, y_train, flagvec*t.from_numpy(lossweight))
        loss.backward()
        optimizer.step()
        tq.set_description("epoch %i loss %f" % (epoch, loss.item()))
        tq.refresh() # to show immediately the update
        trainloss += loss.item()
        del loss, x_train, y_train, o
    testtq = tqdm(testdataloader, desc='test loss', leave=True)
    testloss = [0 for _ in range(9)]
    for x_test, y_test, _ in testtq:
#         print(x_test.numpy().shape)
        with t.no_grad():
            x_test = t.autograd.Variable(x_test.cuda())
#             y_test = t.autograd.Variable(y_test.cuda())
        o = model(x_test)
        loss = caldice(o, y_test)
        testtq.set_description("epoch %i test loss %f" % (epoch, sum(loss)/9))
        testtq.refresh() # to show immediately the update
        testloss = [l+tl for l,tl in zip(loss, testloss)]
        del x_test, y_test, o
    testloss = [l / len(testtq) for l in testloss]
    for cls in range(9):
        if maxloss[cls] < testloss[cls]:
            maxloss[cls] = testloss[cls]
            t.save(model, savename+str(cls+1))
    print('epoch %i TRAIN loss %.4f' % (epoch, trainloss/len(tq)))
    print('test loss %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % tuple(testloss))
    print('best test loss %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % tuple(maxloss))
    if epoch % 10 == 0:
        testloss = [0 for _ in range(9)]
        ntest = [0 for _ in range(9)]
        testtq = tqdm(traindataloader, desc='loss', leave=True)
        for x_test, y_test, _ in testtq:
    #         print(x_test.numpy().shape)
            with t.no_grad():
                x_test = t.autograd.Variable(x_test.cuda())
#                 y_test = t.autograd.Variable(y_test.cuda())
            o = model(x_test)
            loss = caldice(o, y_test)
            testtq.set_description("epoch %i test loss %f" % (epoch, sum(loss)/9))
            testtq.refresh() # to show immediately the update
            testloss = [l+tl if l != -1 else tl for l,tl in zip(loss, testloss)]
            ntest = [n+1 if l != -1 else n for l, n in zip(loss, ntest)]
            del x_test, y_test, o
        testloss = [l / n for l,n in zip(testloss, ntest)]
        print('train loss %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % tuple(testloss))