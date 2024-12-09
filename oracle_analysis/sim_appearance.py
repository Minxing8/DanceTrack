# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import os
# import glob as gb
# import numpy as np
# import cv2
# import torch
# import argparse
# from deepsort_tracker.reid_model import Extractor
# from sklearn.manifold import TSNE
# from matplotlib import pyplot as plt

# class AppearanceFeature(object):
#     def __init__(self, model_path, use_cuda=True):

#         self.extractor = Extractor(model_path, use_cuda=use_cuda)
    
#     def update(self, output_results, img_file_name):
#         ori_img = cv2.imread(img_file_name)
#         self.height, self.width = ori_img.shape[:2]
        
#         bboxes = output_results[:, :4]  # x1y1x2y2
#         bbox_xyxy = bboxes
#         bbox_tlwh = self._xyxy_to_tlwh_array(bbox_xyxy)

#         # generate detections
#         features = self._get_features(bbox_tlwh, ori_img)    
        
#         return features
    
#     @staticmethod
#     def _xyxy_to_tlwh_array(bbox_xyxy):
#         if isinstance(bbox_xyxy, np.ndarray):
#             bbox_tlwh = bbox_xyxy.copy()
#         elif isinstance(bbox_xyxy, torch.Tensor):
#             bbox_tlwh = bbox_xyxy.clone()
#         bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
#         bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
#         return bbox_tlwh    
    
#     def _tlwh_to_xyxy(self, bbox_tlwh):
#         """
#         TODO:
#             Convert bbox from xtl_ytl_w_h to xc_yc_w_h
#         Thanks JieChen91@github.com for reporting this bug!
#         """
#         x, y, w, h = bbox_tlwh
#         x1 = max(int(x), 0)
#         x2 = min(int(x+w), self.width - 1)
#         y1 = max(int(y), 0)
#         y2 = min(int(y+h), self.height - 1)
#         return x1, y1, x2, y2
    
#     def _xyxy_to_tlwh(self, bbox_xyxy):
#         x1, y1, x2, y2 = bbox_xyxy

#         t = x1
#         l = y1
#         w = int(x2 - x1)
#         h = int(y2 - y1)
#         return t, l, w, h

#     def _get_features(self, bbox_xywh, ori_img):
#         im_crops = []
#         for box in bbox_xywh:
#             x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
#             im = ori_img[y1:y2, x1:x2]
#             im_crops.append(im)
#         if im_crops:
#             features = self.extractor(im_crops)
#             features = np.asarray(features) / np.linalg.norm(features, axis=1, keepdims=True)
#         else:
#             features = np.array([])
#         return features    
    
    
    
# parser = argparse.ArgumentParser()
# parser.add_argument('--start', default=0, type=int)
# parser.add_argument('--end', default=8, type=int)
# args = parser.parse_args()

# # dataset = 'mot/val'
# # dataset = '/home/minxing/datasets/NSVA_157_person/test'
# dataset = '/home/minxing/datasets/sportsmot_publish/sportsmot_publish/dataset/val'

# # val_pred = 'val_appearance'
# # if not os.path.exists(val_pred):
# #     os.makedirs(val_pred)
    
# video_cosine_dist_ret = []
# val_seqs = sorted(os.listdir(dataset))[args.start:args.end+1]
# # val_seqs = sorted(os.listdir(dataset))
# for video_name in val_seqs:
#     print(video_name)
#     det_results = {}
#     with open(os.path.join(dataset, video_name, 'gt/gt.txt'), 'r') as f:
#         for line in f.readlines():
#             linelist = line.split(',')
#             img_id = linelist[0]
#             bbox = [float(linelist[2]), 
#                     float(linelist[3]),
#                     float(linelist[2]) + float(linelist[4]),
#                     float(linelist[3]) + float(linelist[5]), 
#                     float(linelist[1])]
#             if int(linelist[7]) == 1:
#                 if int(img_id) in det_results:
#                     det_results[int(img_id)].append(bbox)
#                 else:
#                     det_results[int(img_id)] = list()
#                     det_results[int(img_id)].append(bbox)
#     f.close()
    
#     cosine_dist_ret = []
#     # star_idx = len(gb.glob(os.path.join(dataset, video_name, 'img1') + "/*.jpg")) // 2 + 1
#     tracker = AppearanceFeature(model_path='ckpt.t7')
#     for frame_id in sorted(det_results.keys()):
#         dets = det_results[frame_id]
#         dets = np.array(dets)
#         # image_path = os.path.join(dataset, video_name, 'img1', '{:0>6d}.jpg'.format(frame_id + star_idx))
#         image_path = os.path.join(dataset, video_name, 'img1', '{:0>6d}.jpg'.format(frame_id))
#         # image_path = os.path.join(dataset, video_name, 'img1', '{:0>8d}.jpg'.format(frame_id))
        
#         appearance_feat = tracker.update(dets, image_path)
        
#         cosine_dist_mat = 1. - np.dot(appearance_feat, appearance_feat.T)
#         cosine_dist  = cosine_dist_mat.sum() / len(appearance_feat) / len(appearance_feat)
#         cosine_dist_ret.append(cosine_dist)
    
#     video_cosine_dist_ret.append(sum(cosine_dist_ret) / len(cosine_dist_ret))
#     print(video_cosine_dist_ret)


    
import matplotlib.pyplot as plt
import numpy as np


sportsmot = [0.23203786479945712, 0.22979519059495015, 0.22521789426540542, 0.23383251747983874, 0.23058030395130766, 
             0.23499791608907986, 0.25119243966915444, 0.24945774778928603, 0.24836457493979866]
mot  = [0.289, 0.327, 0.240, 0.224, 0.301, 0.262, 0.269]
# dancetrack = [0.173, 0.150, 0.181, 0.181, 0.216, 0.176, 0.186, 0.215, 0.227, 0.181, 
#               0.214, 0.172, 0.206, 0.204, 0.200, 0.236, 0.176, 0.172, 0.221, 0.170,
#               0.212, 0.233, 0.207, 0.229, 0.140]
# nsva = [0.2502410975665071, 0.24747616088828042, 0.2429175785311062, 0.2504677916327666, 0.23458416090360162, 0.251283568100448, 0.22306200348401112, 
# 0.2515134374012268, 0.24222356969347367, 0.23516706131771495, 0.22459592285964478, 0.2261391434284493, 0.13981967926000102, 0.22577867394370382, 
# 0.19486665576734702, 0.12696189259309504, 0.19492964580597893, 0.20260210055138, 0.24778265832503304, 0.2451456421841318, 0.23466533479246546, 
# 0.2378460702376823, 0.22880600205405932, 0.24087998270584082, 0.2110360812479499, 0.22158414775471527, 0.2138421941478315, 0.23522385892461717, 
# 0.2377637157245335, 0.23715363140919096, 0.18407965600680645, 0.19971680927569396, 0.20313619890739693, 0.226154186759625, 0.21783141079284205, 
# 0.25405454850556575, 0.243969890513982, 0.23721197016526296, 0.23542064388251854, 0.23818956934126953, 0.21626375360091388, 0.24084806104277973, 
# 0.24143852149642728, 0.2539242489105377, 0.2389694573846253, 0.22955594562122222, 0.12294322036426358, 0.24071685895493808]
nsva = [0.23458416090360162, 0.251283568100448, 0.22306200348401112, 0.22459592285964478, 0.2261391434284493, 0.13981967926000102, 0.22577867394370382, 
        0.19486665576734702, 0.12696189259309504, 0.19492964580597893, 0.20260210055138]

mot_mean = np.mean(mot)
sportsmot_mean = np.mean(sportsmot)
nsva_mean = np.mean(nsva)

mot_x = range(len(mot))
sportsmot_x = range(len(mot), len(sportsmot) + len(mot))
# dancetrack_x = range(len(mot), len(mot) + len(dancetrack))
nsva_x = range(len(sportsmot) + len(mot), len(sportsmot) + len(mot) + len(nsva))

fig, ax = plt.subplots(figsize=(15, 5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.bar(x=mot_x, height=mot, alpha=0.3, color='blue', label='MOT17')
plt.bar(x=sportsmot_x, height=sportsmot, alpha=0.3, color='green', label='SportsMOT')
# plt.bar(x=dancetrack_x, height=dancetrack, alpha=0.3, color='red', label='DanceTrack')
plt.bar(x=nsva_x, height=nsva, alpha=0.3, color='red', label='NSVA Track')

plt.plot([mot_x[0], mot_x[-1]], [mot_mean, mot_mean], linestyle='--', color='blue', linewidth=2)  # MOT17
plt.plot([sportsmot_x[0], sportsmot_x[-1]], [sportsmot_mean, sportsmot_mean], linestyle='--', color='green', linewidth=2)  # SPORTSMOT
plt.plot([nsva_x[0], nsva_x[-1]], [nsva_mean, nsva_mean], linestyle='--', color='red', linewidth=2)  # NSVA

plt.legend(fontsize=16)
plt.xticks([])
plt.ylim((0.10, 0.35))
plt.title("Cosine distance of re-ID feature", fontsize=16)
plt.savefig('reid_sim_sportsmot_nsva_mot.png', bbox_inches='tight', dpi=100)
plt.close()
