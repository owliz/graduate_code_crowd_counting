
# Borrowed from https://github.com/leeyeehoo/CSRNet-pytorch
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from tqdm import tqdm
# from numba import cuda
#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
# @cuda.autojit()
def gaussian_filter_density(gt):
#     print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(np.c_[np.nonzero(gt)[1], np.nonzero(gt)[0]])
    
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

#     print ('generate density...')
    for i, pt in (enumerate(pts)):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
#     print ('done.')
    return density


#set the root to the Shanghai dataset you download
root = '/home/orli/Blue-HDD/1_final_lab_/Dataset/crowd_conuting/CSRNet/' +\
       'ShanghaiTech_Crowd_Counting_Dataset/'

part_A_train = os.path.join(root,'part_A/train_data','images')
part_A_test = os.path.join(root,'part_A/test_data','images')
part_B_train = os.path.join(root,'part_B/train_data','images')
part_B_test = os.path.join(root,'part_B/test_data','images')

#now generate the ShanghaiA's ground truth
path_sets = [part_A_train, part_A_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for path in path_sets:
    lab_root_path = path.replace('images', 'labels')
    if not os.path.exists(lab_root_path):
        os.makedirs(lab_root_path)
    gt_h5_root_path = path.replace('images', 'ground_truth_h5')
    if not os.path.exists(gt_h5_root_path):
        os.makedirs(gt_h5_root_path)

# for img_path in tqdm(img_paths):
# #     print (img_path)
#     mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace(
#         'IMG_','GT_IMG_'))
#     lab_path = img_path.replace('.jpg','.npy').replace('images','labels').replace('IMG_','LAB_')
#     img= plt.imread(img_path)
#     k = np.zeros((img.shape[0],img.shape[1]))
#     gt = mat["image_info"][0,0][0,0][0]
#     for i in range(0, len(gt)):
#         if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
#             k[int(gt[i][1]), int(gt[i][0])] = 1
#     k = gaussian_filter_density(k)
#     with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth_h5'), 'w') as hf:
#         hf['density'] = k
#
#     if not os.path.exists(path=lab_path):
#         np.save(lab_path, k)
#         plt.imshow(k, cmap='jet')
#         plt.show()



#now see a sample from ShanghaiA
plt.imshow(Image.open(img_paths[0]))

gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth_h5'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
print(np.sum(groundtruth)) # don't mind this slight variation


#now generate the ShanghaiB's ground truth
path_sets = [part_B_train, part_B_test]

for path in path_sets:
    # lab_root_path = path.replace('images', 'labels')
    # if not os.path.exists(lab_root_path):
    #     os.makedirs(lab_root_path)
    gt_h5_root_path = path.replace('images', 'ground_truth_h5')
    if not os.path.exists(gt_h5_root_path):
        os.makedirs(gt_h5_root_path)

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

# for img_path in tqdm(img_paths):
#     # print(img_path)
#     mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace(
#         'IMG_','GT_IMG_'))
#     img= plt.imread(img_path)
#     k = np.zeros((img.shape[0],img.shape[1]))
#     gt = mat["image_info"][0,0][0,0][0]
#     for i in range(0,len(gt)):
#         if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
#             k[int(gt[i][1]),int(gt[i][0])]=1
#     k = gaussian_filter(k,15)
#     file_path = img_path.replace('.jpg','.h5').replace('images','ground_truth_h5')
#     with h5py.File(file_path, 'w') as hf:
#             hf['density'] = k

# #now see a sample from ShanghaiB
file_path = img_paths[22].replace('.jpg','.h5').replace('images','ground_truth_h5')
gt_file = h5py.File(file_path, 'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth, cmap=CM.jet)
print("Sum = ", np.sum(groundtruth))

# Image corresponding to the ground truth
img = Image.open(file_path.replace('.h5','.jpg').replace('ground_truth_h5','images'))
plt.imshow(img)
print(file_path.replace('.h5','.jpg').replace('ground_truth_h5','images'))

