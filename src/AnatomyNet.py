
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

TRAIN_PATH = './data/trainpddca15_crp_v2_pool1.pth'
TEST_PATH = './data/testpddca15_crp_v2_pool1.pth'
CET_PATH = './data/trainpddca15_cet_crp_v2_pool1.pth'
PET_PATH = './data/trainpddca15_pet_crp_v2_pool1.pth'
os.environ["CUDA_VISIBLE_DEVICES"]="7"



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
def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride       
    def forward(self, x):
#         print(x.size())
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
#         print(x.size(), residual.size(), out.size())
        out += residual
        out = self.relu(out)
        return out
def Deconv3x3x3(in_planes, out_planes, stride=2):
    "3x3x3 deconvolution with padding"
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=stride)

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=15):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.LeakyReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y
class SEBasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=15):
        super(SEBasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer3D(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
#         if self.downsample is not None:
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class UpSEBasicBlock3D(nn.Module):
    def __init__(self, inplanes1, inplanes2, planes, stride=1, downsample=None, reduction=16):
        super(UpSEBasicBlock3D, self).__init__()
        inplanes3 = inplanes1 + inplanes2
        if stride == 2:
            self.deconv1 = Deconv3x3x3(inplanes1, inplanes1//2)
            inplanes3 = inplanes1 // 2 + inplanes2
        self.stride = stride
        # self.conv1x1x1 = nn.Conv3d(inplanes2, planes, kernel_size=1, stride=1)#, padding=1)
        self.conv1 = conv3x3x3(inplanes3, planes)#, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer3D(planes, reduction)
        if inplanes3 != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes3, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x1, x2):
#         print(x1.size(), x2.size())
        if self.stride == 2: x1 = self.deconv1(x1)
        # x2 = self.conv1x1x1(x2)
        #print(x1.size(), x2.size())
        out = t.cat([x1, x2], dim=1) #x1 + x2
        residual = self.downsample(out)
        #print(residual.size(), x1.size(), x2.size())
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        #print(out.size(), residual.size())
        out += residual
        out = self.relu(out)
        return out
class UpBasicBlock3D(nn.Module):
    def __init__(self, inplanes1, inplanes2, planes, stride=2):
        super(UpBasicBlock3D, self).__init__()
        inplanes3 = inplanes1 + inplanes2
        if stride == 2:
            self.deconv1 = Deconv3x3x3(inplanes1, inplanes1//2)
            inplanes3 = inplanes1//2 + inplanes2
        self.stride = stride
        # elif inplanes1 != planes:
            # self.deconv1 = nn.Conv3d(inplanes1, planes, kernel_size=1, stride=1)
        # self.conv1x1x1 = nn.Conv3d(inplanes2, planes, kernel_size=1, stride=1)#, padding=1)
        self.conv1 = conv3x3x3(inplanes3, planes)#, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        if inplanes3 != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes3, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x1, x2):
#         print(x1.size(), x2.size())
        if self.stride == 2: x1 = self.deconv1(x1)
        #print(self.stride, x1.size(), x2.size())
        out = t.cat([x1, x2], dim=1)
        residual = self.downsample(out)
        #print(out.size(), residual.size())
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out
class ResNetUNET3D(nn.Module):
    def __init__(self, block, upblock, upblock1, n_size, num_classes=2, in_channel=1): # BasicBlock, 3
        super(ResNetUNET3D, self).__init__()
        self.inplane = 28
        self.conv1 = nn.Conv3d(in_channel, self.inplane, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, 30, blocks=n_size, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=1)
        self.layer3 = self._make_layer(block, 34, blocks=n_size, stride=1)
        self.layer4 = upblock(34, 32, 32, stride=1)
        self.inplane = 32
        self.layer5 = self._make_layer(block, 32, blocks=n_size-1, stride=1)
        self.layer6 = upblock(32, 30, 30, stride=1)
        self.inplane = 30
        self.layer7 = self._make_layer(block, 30, blocks=n_size-1, stride=1)
        self.layer8 = upblock(30, 28, 28, stride=1)
        self.inplane = 28
        self.layer9 = self._make_layer(block, 28, blocks=n_size-1, stride=1)
        self.inplane = 28
        self.layer10 = upblock1(28, 1, 14, stride=2)
        self.layer11 = nn.Sequential(#nn.Conv3d(16, 14, kernel_size=3, stride=1, padding=1, bias=True),
                                     #nn.ReLU(inplace=True),
                                     nn.Conv3d(14, num_classes, kernel_size=3, stride=1, padding=1, bias=True))
#         self.outconv = nn.ConvTranspose3d(self.inplane, num_classes, 2, stride=2)
        self.initialize()
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)
    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes
        return nn.Sequential(*layers)
    def forward(self, x0):
        x = self.conv1(x0) # 16 1/2 
        x = self.bn1(x)
        x1 = self.relu(x)

        x2 = self.layer1(x1) # 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4
        x3 = self.layer2(x2) # 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8
        x4 = self.layer3(x3) # 64 1/16 64 1/16 res 64 1/16 - 64 1/16 64 1/16 res 64 1/16 - 64 1/16 64 1/16 res 64 1/16
#         print('x4', x4.size())
        x5 = self.layer4(x4, x3) # 16 1/8 48 1/8 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8
        x5 = self.layer5(x5)
        x6 = self.layer6(x5, x2) # 8 1/4 24 1/4 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4
        x6 = self.layer7(x6)
        x7 = self.layer8(x6, x1) # 4 1/2 20 1/2 16 1/2 16 1/2 res 16 1/2 - 16 1/2 16 1/2 res 16 1/2 - 16 1/2 16 1/2 res 16 1/2
        x7 = self.layer9(x7)
        x8 = self.layer10(x7, x0)
        x9 = self.layer11(x8)
#         print(x0.size(), x.size(), x1.size(), x2.size(), x3.size(), x4.size(), x5.size(), x6.size(), \
#               x7.size(), x8.size(), x9.size())
        return F.softmax(x9, dim=1)
#         out = self.outconv(x7)
#         return F.softmax(out, dim=1)
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


def focal(y_pred, y_true, flagvec):
    retv = - t.sum(t.sum(t.sum(t.sum(t.log(t.clamp(y_pred,1e-6,1))*y_true.type(t.cuda.FloatTensor)*t.pow(1-y_pred,2),4),3),2),0) 
        * flagvec.cuda() 


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
model = ResNetUNET3D(SEBasicBlock3D, UpSEBasicBlock3D, UpBasicBlock3D, 2, num_classes=9+1, in_channel=1).cuda() 
lossweight = np.array([2.22, 1.31, 1.99, 1.13, 1.93, 1.93, 1.0, 1.0, 1.90, 1.98], np.float32)
pretraind_dict = t.load('./model/unet10pool3e2e_seres18_conc_pet_wmask_2_rmsp_1')["weight"]
model_dict = model.state_dict()
pretraind_dict = {k: v for k, v in pretraind_dict.items() if k in model_dict}
model_dict.update(pretraind_dict)
model.load_state_dict(pretraind_dict)
savename = './model/unet10pool3e2e_seres18_conc_pet_wmask_2_rmsp_lru_1_'


# In[5]:


optimizer = t.optim.RMSprop(model.parameters(),lr = 5e-4)
maxloss = [0 for _ in range(9)]
for epoch in range(150):
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
            state = {"epoch": epoch, "weight": model.state_dict()}
            t.save(state, savename+str(cls+1))
#             model.load_state_dict(t.load(savename)["weight"])
#             t.save(model, savename+str(cls+1))
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


optimizer = t.optim.SGD(model.parameters(), 1e-4, momentum = 0.9)#, weight_decay = 1e-4)
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
            state = {"epoch": epoch, "weight": model.state_dict()}
            t.save(state, savename+str(cls+1))
#             model.load_state_dict(t.load(savename)["weight"])
#             t.save(model, savename+str(cls+1))
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


