import numpy as np
import os 
from typing import Iterable, Optional, Tuple
from scipy import linalg

from features.kinetic import extract_kinetic_features
from features.manual import extract_manual_features
import pdb
from tqdm import tqdm
import torch

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    Code apapted from https://github.com/mseitzer/pytorch-fid
    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    mu and sigma are calculated through:
    ```
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    ```
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)
    # pdb.set_trace()
    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std
    
    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist


def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    print(feature_list.shape)
    # normalize the scale
    mean = np.mean(feature_list, axis=0)
    std = np.std(feature_list, axis=0) + 1e-10
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import json
music_root='./dataset/music_feature/'
def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        sample_dict = json.loads(f.read())
        # pdb.set_trace()
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, -1][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, -1]


        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
        
        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')


        # print(len(beats))
        return beat_axis


def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def calc_ba_score(root):

    # gt_list = []
    ba_scores = []
    bbk,bpop,block,bmp,blp,bho,bwa,bkr,bsj,bbj=[],[],[],[],[],[],[],[],[],[]
    for pkl in os.listdir(root):
        # print(pkl)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pose_decode'][:, :]

        dance_beats, length = calc_db(joint3d, pkl)   
        name = np.load(os.path.join(root, pkl), allow_pickle=True).item()['leader_name']
             
        music_beats = get_mb(name.split('_')[-3] + '.json', length)

        gen = name.split('_')[0]
        if gen == 'gBR':
            bbk.append(BA(music_beats, dance_beats))
        elif gen == 'gPO':
            bpop.append(BA(music_beats, dance_beats))
        elif gen == 'gLO':
            block.append(BA(music_beats, dance_beats))
        elif gen == 'gMH':
            bmp.append(BA(music_beats, dance_beats))
        elif gen == 'gLH':
            blp.append(BA(music_beats, dance_beats))
        elif gen == 'gHO':
            bho.append(BA(music_beats, dance_beats))
        elif gen == 'gWA':
            bwa.append(BA(music_beats, dance_beats))
        elif gen == 'gKR':
            bkr.append(BA(music_beats, dance_beats))
        elif gen == 'gJS':
            bsj.append(BA(music_beats, dance_beats))
        elif gen == 'gJB':
            bbj.append(BA(music_beats, dance_beats))
        else:
            print('nan')

        # ba_scores.append(BA(music_beats, dance_beats))
    
    print('bbk',np.mean(bbk))
    print('bpop',np.mean(bpop))
    print('block',np.mean(block))
    print('bmp',np.mean(bmp))
    print('blp',np.mean(blp))
    print('bho',np.mean(bho))
    print('bwa',np.mean(bwa))
    print('bkr',np.mean(bkr))
    print('bsj',np.mean(bsj))
    print('bbj',np.mean(bbj))
    return np.mean(bbk)

def main():
    # gt_path2='/yaosiyue/diff_vqvae/s0.5/'
    # gt_path='/yaosiyue/diff_vqvae/exp_ckpt/eval/final/gs9_s0.6_a0.9_wo2/ep000200/NoiseStep50/'
    gt_path='./exp_ckpt/eval/final/s0.5_final/ep000200//NoiseStep50/'
    bs = calc_ba_score(gt_path)
    # print(bs)
    pdb.set_trace()
    gt_file=os.listdir(gt_path)
    print(gt_path)
    # result_path='./result/'
    # result_file=os.listdir(result_path)
    fid_main=[]
    fid_part =0
    div_gt=[]
    fbk,fpop,flock,fmp,flp,fho,fwa,fkr,fsj,fbj=[],[],[],[],[],[],[],[],[],[]
    
    dbk,dpop,dlock,dmp,dlp,dho,dwa,dkr,dsj,dbj=[],[],[],[],[],[],[],[],[],[]

    for gt in tqdm(gt_file):
        # print('gt')
        # keypoints3d2 = np.load(gt_path2+gt[:-4]+'.pkl',allow_pickle=True) #.item()
        # print(keypoints3d)
        # pdb.set_trace()
        keypoints3d = np.load(gt_path+gt,allow_pickle=True).item()
        # keypoints3d_gt = keypoints3d['real_decode']
        keypoints3d_res = keypoints3d['pose_decode']
        # keypoints3d_m = keypoints3d['main_decode']
        # keypoints3d2 = np.load(gt_path2+gt,allow_pickle=True).item()
        keypoints3d_gt = keypoints3d['real_decode']
        gen = keypoints3d['leader_name'].split('_')[0]
        # mask=keypoints3d['mask'].cpu().numpy()
        # coder= keypoints3d['real_code'].cpu().numpy()
        # codep= keypoints3d['pred_code'].cpu().numpy()
        # inn = np.nonzero(coder*mask==codep*mask)[1]
        # pdb.set_trace()
        # keypoints3d_res = keypoints3d2['pred'].cpu().numpy().reshape(-1,72)
        # keypoints3d_gt = keypoints3d['gt'].cpu().numpy().reshape(-1,72)
        # keypoints3d_m = keypoints3d['main'].cpu().numpy().reshape(-1,72)
        mask=keypoints3d['mask']
        size=mask[0][0][:32].repeat(len(mask)).cpu().numpy()
        idx=[]
        bidx=[]
        for i in range(len(size)):
            if size[i]==1:
                idx.extend([id for id in range(i*8,i*8+8)])
            else:
                bidx.extend([id for id in range(i*8,i*8+8)])
        # print(len(inn)*8/len(idx))
        # pdb.set_trace()
        
        
        root = keypoints3d_gt[:, :3]  # the root
        keypoints3d_gt = keypoints3d_gt - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
        keypoints3d_gt[:, :3] = 0
        
        root = keypoints3d_res[:, :3]  # the root
        keypoints3d_res = keypoints3d_res - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
        keypoints3d_res[:, :3] = 0

        # root = keypoints3d_m[:, :3]  # the root
        # keypoints3d_m = keypoints3d_m - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
        # keypoints3d_m[:, :3] = 0
        
        # kinetic_gt= extract_kinetic_features(keypoints3d_gt.reshape(-1,24,3))
        # manual_gt = extract_manual_features(keypoints3d_gt.reshape(-1,24,3))
        # # # keypoints3d_res = np.load(result_path+gt,allow_pickle=True)
        # kinetic_res = extract_kinetic_features(keypoints3d_res.reshape(-1,24,3))
        # manual_res = extract_manual_features(keypoints3d_res.reshape(-1,24,3))
        ##
        # diff = torch.nn.MSELoss()
        # d = diff(torch.tensor(keypoints3d_m[idx]),torch.tensor(keypoints3d_res[idx]))
        # print(d)
        # ###caculate FID
        # fid_kin.append(calculate_frechet_feature_distance(kinetic_gt,kinetic_res))
        # fid_man.append( calculate_frechet_feature_distance(manual_gt,manual_res))
        # pdb.set_trace()
        # fid_main.append(calculate_frechet_feature_distance(keypoints3d_gt,keypoints3d_res))
        fid = calculate_frechet_feature_distance(keypoints3d_gt,keypoints3d_res)
        # fid_part += calculate_frechet_feature_distance(keypoints3d_gt[bidx],keypoints3d_res[bidx])

        # fid_k = calculate_frechet_feature_distance(kinetic_gt[bidx],kinetic_res[bidx])
        # fid_m = calculate_frechet_feature_distance(manual_gt[bidx],manual_res[bidx])

        ###caculate DIV
        # key = keypoints3d_m[bidx] - keypoints3d_res[bidx]
        # div_gt.append(calculate_avg_distance(keypoints3d_res))
        div =calculate_avg_distance(keypoints3d_res)
        # div_res.append(calculate_avg_distance(keypoints3d_res))
        # div_gt += calculate_avg_distance(keypoints3d_gt)
        # div_res += calculate_avg_distance(key[idx])
        # pdb.set_trace()

        if gen == 'gBR':
            fbk.append(fid)
            dbk.append(div)
        elif gen == 'gPO':
            fpop.append(fid)
            dpop.append(div)
        elif gen == 'gLO':
            flock.append(fid)
            dlock.append(div)
        elif gen == 'gMH':
            fmp.append(fid)
            dmp.append(div)
        elif gen == 'gLH':
            flp.append(fid)
            dlp.append(div)
        elif gen == 'gHO':
            fho.append(fid)
            dho.append(div)
        elif gen == 'gWA':
            fwa.append(fid)
            dwa.append(div)
        elif gen == 'gKR':
            fkr.append(fid)
            dkr.append(div)
        elif gen == 'gJS':
            fsj.append(fid)
            dsj.append(div)
        elif gen == 'gJB':
            fbj.append(fid)
            dbj.append(div)
        else:
            print('nan')

    # print(np.mean(fid_main))
    # print(np.mean(div_gt))
    # print(np.mean(fid_kin))
    # print(np.mean(fid_man))
    # print(np.mean(div_res))
    pdb.set_trace()
    t_fid_kin = fid_main/len(gt_file)
    t_fid_man = fid_part/len(gt_file)
    t_div_kin = div_gt/len(gt_file)
    # t_div_man = div_res/len(gt_file)
    
    print('fid_main',t_fid_kin)
    print('fid_part',fid_part)
    print('div_gt',t_div_kin)
    # print('div_res',t_div_man)

    pkl_data = {'t_fid_kin':t_fid_kin,'t_fid_man':t_fid_man,'t_div_kin':t_div_kin} #,'t_div_man':t_div_man}
    ep_path = os.path.join('./evaluate_result/')
    if not os.path.exists(ep_path):
        os.makedirs(ep_path)
    np.save(ep_path, pkl_data)


if __name__ == '__main__':
    main()
