import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import json
import pdb
import pickle
from tqdm import tqdm
import itertools
import random
from vqvae.log import Logger

import diffusion.gaussian_diffusion as gd
from diffusion.gaussian_diffusion import get_named_beta_schedule, GaussianDiffusion
import diffusion.unet as model_unet
from torch.optim import Adam
from torch.optim import SGD
import vqvae.sep_vqvae as model_vqvae
from diffusion.unet import UNetModel
from diffusion.nn import linear
from torch.utils.data.distributed import DistributedSampler
import os


class MotionDataset(Dataset):
    def __init__(self, musics, dances1, dances2=None, name=None):
        if dances2 is not None:
            assert (len(musics) == len(dances1) == len(dances2)), \
                'the number of dances should be equal to the number of musics'
        
        self.musics = musics
        self.dances1 = dances1
        self.dances2 = dances2
        self.name = name

    def __len__(self):
        return len(self.musics)

    def __getitem__(self, index):
        if self.dances2 is not None:
            if self.name is not None:
                return self.musics[index], self.dances1[index], self.dances2[index], self.name[index]
            else:
                return self.musics[index], self.dances1[index], self.dances2[index]
        else:
            return self.musics[index], self.dances1[index]

        
class DiffVQVAE():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()
    
    def _build(self):
        config = self.config
        # self.local_rank = int(os.environ["LOCAL_RANK"])
        self._build_model()
        if not(hasattr(config, 'need_not_train_data') and config.need_not_train_data):
            self._build_train_loader()
        if not(hasattr(config, 'need_not_test_data') and config.need_not_test_data):      
            self._build_test_loader()
        self._build_optimizer()
        self._build_diffusion_model()
        
    def _build_model(self):
        """ Define Model """
        config = self.config 

        # torch.distributed.init_process_group(backend="nccl")
        # # self.local_rank = torch.distributed.get_rank()
        # # torch.cuda.set_device(local_rank)
        
        # device = torch.device("cuda", self.local_rank)


        if hasattr(config.structure_vqvae, 'name') and hasattr(config.structure_unet_fill, 'name') and hasattr(config.structure_unet_recode, 'name'):
            print(f'using {config.structure_vqvae.name} and {config.structure_unet_fill.name} ')
            
            model_classv = getattr(model_vqvae, config.structure_vqvae.name)
            modelv = model_classv(config.structure_vqvae)

            model_classu = getattr(model_unet, config.structure_unet_recode.name)
            modelu = model_classu(config.structure_unet_recode)
            
            model_classf = getattr(model_unet, config.structure_unet_fill.name)
            modelf = model_classf(config.structure_unet_fill)
            
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        
        device_ids = [0,]
        # self.modelv = modelv.to(device)
        # self.modelu = modelu.to(device)
        # self.modelf = modelf.to(device)
        # self.modelv=nn.parallel.DistributedDataParallel(self.modelv)
        # self.modelu=nn.parallel.DistributedDataParallel(self.modelu)
        # self.modelf=nn.parallel.DistributedDataParallel(self.modelf)
        
        modelv = nn.DataParallel(modelv, device_ids=device_ids)
        modelu = nn.DataParallel(modelu, device_ids=device_ids)
        modelf = nn.DataParallel(modelf, device_ids=device_ids)
        self.modelv = modelv.cuda()
        self.modelu = modelu.cuda()
        self.modelf = modelf.cuda()
        
    def _build_diffusion_model(self):
        #init diffusion model fill
        self.timesteps_f = self.config.structure_diffusion_fill.timesteps
        beats=get_named_beta_schedule('linear',self.timesteps_f)
        if hasattr(self.config.structure_diffusion_fill, 'model_mean_type'):
            model_mean_type = getattr(gd.ModelMeanType, self.config.structure_diffusion_fill.model_mean_type)
        if hasattr(self.config.structure_diffusion_fill, 'model_var_type'):
            model_var_type = getattr(gd.ModelVarType, self.config.structure_diffusion_fill.model_var_type)
        if hasattr(self.config.structure_diffusion_fill, 'loss_type'):
            loss_type = getattr(gd.LossType, self.config.structure_diffusion_fill.loss_type)
            
        self.diffusion_fill = GaussianDiffusion(betas=beats, model_mean_type= model_mean_type, model_var_type= model_var_type, loss_type= loss_type)
    
        #init diffusion model recode
        self.timesteps_r = self.config.structure_diffusion_recode.timesteps
        beats=get_named_beta_schedule('linear',self.timesteps_r)
        if hasattr(self.config.structure_diffusion_recode, 'model_mean_type'):
            model_mean_type = getattr(gd.ModelMeanType, self.config.structure_diffusion_recode.model_mean_type)
        if hasattr(self.config.structure_diffusion_recode, 'model_var_type'):
            model_var_type = getattr(gd.ModelVarType, self.config.structure_diffusion_recode.model_var_type)
        if hasattr(self.config.structure_diffusion_recode, 'loss_type'):
            loss_type = getattr(gd.LossType, self.config.structure_diffusion_recode.loss_type)
        
        self.diffusion_recode = GaussianDiffusion(betas=beats, model_mean_type= model_mean_type, model_var_type= model_var_type, loss_type= loss_type)
        
        
        
    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer_f = optim(self.modelf.parameters(), **config.kwargs)
        self.schedular_f = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_f, **config.schedular_kwargs)
        
        self.optimizer_r = optim(self.modelu.parameters(), **config.kwargs)
        self.schedular_r = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_r, **config.schedular_kwargs)
        
    def _build_test_loader(self):
        data=self.config.data
        music_data, dance_data_mian, dance_data_back = [], [], []
        name = []
        fnames = sorted(os.listdir(data.train_dir))
        # print(fnames)
        if ".ipynb_checkpoints" in fnames:
            fnames.remove(".ipynb_checkpoints")
        print('**********',data.train_dir)
        self.test_name=[]
        for fname in fnames[200:250]:
            self.test_name.append(fname)
            path = os.path.join(data.train_dir, fname)
            dance = np.load(path,allow_pickle=True)
            # np_dance=np_dance['smpl_poses']  #(lenght,72)
            np_dance=dance['lead_dancer'].reshape(-1,72)  #(lenght,72)
            back_dance=dance['partner_dancer'].reshape(-1,72)
            #pdb.set_trace()
            # music_name=fname.split('_')[6]   
            music_name=dance['leader_name'].split('_')[6] 
            name.append(dance['leader_name'])
            mpath = os.path.join(data.music_dir, music_name+'.json')
            with open(mpath) as f:
                music_dict = json.loads(f.read())
                np_music = np.array(music_dict['music_array'])   #(lenght,419)
                f.close()
            
            assert np_music.shape[-1] == 419, 'wrong music dim'
            
            # back_name=fname[:-11]
            
            # if fname.split('_')[-1] == 'people0.pkl':
            #     name=['people1.pkl', 'people2.pkl']
            # elif fname.split('_')[-1] == 'people1.pkl':
            #     name=['people0.pkl', 'people2.pkl']
            # elif fname.split('_')[-1] == 'people2.pkl':
            #     name=['people0.pkl', 'people1.pkl']
            # else:
            #     print('name error!')
            
            # for n in name:     
            #     path = os.path.join(data.train_dir, back_name+n)
            #     if not os.path.exists(path):
            #         print('file not exist')
            #         continue
            #     back_dance = np.load(path,allow_pickle=True)
            #     # back_dance=back_dance['smpl_poses']  #(lenght,72)
                
            if not data.rotmat:
                root = np_dance[:, :3]  # the root
                np_dance = np_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                np_dance[:, :3] = root
                
                root = back_dance[:, :3]  # the root
                back_dance = back_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                back_dance[:, :3] = root
        
        
            interval=data.interval  
            move = data.move
            seq_len=len(np_dance)
            # if len(np_dance) != len(back_dance):
            #     continue
            assert len(np_dance) == len(back_dance), "different motion length"
            # if interval is not None:
            #     for i in range(0, seq_len, move):
            #         music_sub_seq = np_music[i: i + interval]
            #         dance_sub_seq = np_dance[i: i + interval]
            #         bdance_sub_seq = back_dance[i: i + interval]
            #         if len(dance_sub_seq) == interval and len(music_sub_seq) == interval and len(bdance_sub_seq) == interval:
            #             music_data.append(music_sub_seq)
            #             dance_data_mian.append(dance_sub_seq)
            #             dance_data_back.append(bdance_sub_seq)

            # else:
            music_data.append(np_music)
            dance_data_mian.append(np_dance)
            dance_data_back.append(back_dance)
        
        self.testing_data = DataLoader(
            MotionDataset(music_data, dance_data_mian, dance_data_back,name),
            num_workers=32,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            # drop_last=True
        )
    
    def _build_train_loader(self):
        data=self.config.data
        music_data, dance_data_mian, dance_data_back = [], [], []
        fnames = sorted(os.listdir(data.train_dir))
        # print(fnames)
        if ".ipynb_checkpoints" in fnames:
            fnames.remove(".ipynb_checkpoints")
        print('**********',data.train_dir)
        for fname in fnames[:300]:
            path = os.path.join(data.train_dir, fname)
            dance = np.load(path,allow_pickle=True)
            # np_dance=np_dance['smpl_poses']  #(lenght,72)
            np_dance=dance['lead_dancer'].reshape(-1,72)  #(lenght,72)
            back_dance=dance['partner_dancer'].reshape(-1,72)
            #pdb.set_trace()
            # music_name=fname.split('_')[6]   
            music_name=dance['leader_name'].split('_')[6]
            mpath = os.path.join(data.music_dir, music_name+'.json')
            with open(mpath) as f:
                music_dict = json.loads(f.read())
                np_music = np.array(music_dict['music_array'])   #(lenght,419)
                f.close()
            
            assert np_music.shape[-1] == 419, 'wrong music dim'
            
            # back_name=fname[:-11]
            
            # if fname.split('_')[-1] == 'people0.pkl':
            #     name=['people1.pkl', 'people2.pkl']
            # elif fname.split('_')[-1] == 'people1.pkl':
            #     name=['people0.pkl', 'people2.pkl']
            # elif fname.split('_')[-1] == 'people2.pkl':
            #     name=['people0.pkl', 'people1.pkl']
            # else:
            #     print('name error!')
            
            # for n in name:     
            #     path = os.path.join(data.train_dir, back_name+n)
            #     if not os.path.exists(path):
            #         print('file not exist')
            #         continue
            #     back_dance = np.load(path,allow_pickle=True)
            #     # back_dance=back_dance['smpl_poses']  #(lenght,72)
                
            if not data.rotmat:
                root = np_dance[:, :3]  # the root
                np_dance = np_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                np_dance[:, :3] = root
                
                root = back_dance[:, :3]  # the root
                back_dance = back_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                back_dance[:, :3] = root

            
            # if len(np_dance) != len(back_dance):
            #     continue
            interval=data.interval  
            move = data.move
            seq_len=len(np_dance)
            assert len(np_dance) == len(back_dance), "different motion length"
            if interval is not None:
                for i in range(0, seq_len, move):
                    music_sub_seq = np_music[i: i + interval]
                    dance_sub_seq = np_dance[i: i + interval]
                    bdance_sub_seq = back_dance[i: i + interval]
                    if len(dance_sub_seq) == interval and len(music_sub_seq) == interval and len(bdance_sub_seq) == interval:
                        music_data.append(music_sub_seq)
                        dance_data_mian.append(dance_sub_seq)
                        dance_data_back.append(bdance_sub_seq)

            else:
                music_data.append(np_music)
                dance_data_mian.append(np_dance)
                dance_data_back.append(back_dance)
       
        # pdb.set_trace()
        # sampler = DistributedSampler(MotionDataset(music_data, dance_data_mian, dance_data_back),shuffle=True)
        self.training_data = DataLoader(
            MotionDataset(music_data, dance_data_mian, dance_data_back),
            num_workers=32,
            batch_size=self.config.batch_size,
            # batch_size=1,
            shuffle=True,
            pin_memory=False,
            # drop_last=True,
            # sampler=sampler,
        )
    
    def feature_selection(self, control_var, dance_seq, device):  #control_var is a number between 0 and 1
        B, E, T = dance_seq.shape
        
        pose_seq_mask = torch.zeros(dance_seq.shape).to(device)
        same_seq_number = np.round(T//2 * control_var)
        
        idx = np.round(np.linspace(0, (T//2)-1, int(same_seq_number))).astype(int)
        pose_seq_mask[:, :, idx] = 1
        pose_seq_mask[:, :, idx+(T//2)] = 1 
    
        return pose_seq_mask


    def train_fill(self):
        vqvae = self.modelv.eval()
        unet_fill = self.modelf.train()
        
        config = self.config
        data = self.config.data
        training_data = self.training_data 
        # testing_data = self.testing_data
        device = torch.device('cuda' if config.cuda else 'cpu')
        timesteps = self.timesteps_r
        
        log = Logger(self.config, './log/loger/changeinput/')
        
        # load model
        checkpoint = torch.load(config.vqvae_weight, map_location={'cuda:0':'cuda:1'})
        vqvae.load_state_dict(checkpoint['model'], strict=False)
              
        if hasattr(config, 'fill_weight') and config.fill_weight is not None and config.fill_weight is not '': 
            checkpoint = torch.load(config.fill_weight, map_location={'cuda:0':'cuda:1'})
            unet_fill.load_state_dict(checkpoint['model'], strict=False)
            
        
        writer_train = SummaryWriter('./log/fill/train/')
        writer_eval = SummaryWriter('./log/fill/eval/')
        
        
        #train model
        for epoch_i in range(1, config.epoch + 1):
            t_loss = 0
            index = 0
            # torch.cuda.empty_cache()
            log.set_progress(epoch_i, len(training_data),self.optimizer_f)
            for batch_i, batch in enumerate(training_data):
                # if batch_i >= 800:
                #     break
                print('########################################')
                # load music and pose data
                music_seq, pose_seq, back_seq = batch
                
                music_seq = music_seq.to(device).to(torch.float)
                music_seq = music_seq.permute(0,2,1)
                
                pose_seq = pose_seq.to(device)
                back_seq = back_seq.to(device)

                pose_seq[:, :, :3] = 0
                back_seq[:, :, :3] = 0

                self.optimizer_f.zero_grad()
                
                with torch.no_grad():
                    #############
                    # if batch_i==0 and epoch_i==101:
                    codebook_up, codebook_down = vqvae.module.return_codebook()
                    ############# 
                    
                    _, dequants_pred_main = vqvae.module.encode(pose_seq)
                    _, dequants_pred_back = vqvae.module.encode(back_seq)
                    
                    if isinstance(dequants_pred_main, tuple):
                        dequants_main = torch.cat([dequants_pred_main[0][0].clone().detach(), dequants_pred_main[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                        dequants_back = torch.cat([dequants_pred_back[0][0].clone().detach(), dequants_pred_back[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                        
                    else:
                        _, dequants_main = vqvae.module.encode(pose_seq)[0][0].clone().detach()
                        _, dequants_back = vqvae.module.encode(back_seq)[0][0].clone().detach()
                
                # if batch_i==0 and epoch_i==101:
                ##Set codebook
                self.diffusion_fill.set_codebook_var(codebook_up, codebook_down)
                    
                # sample t uniformally for every example in the batch
                t = torch.randint(0, timesteps, (len(dequants_main),), device=device).long()
                
                # control the similarity between dancer
                s = random.uniform(0,1)
                pose_mask = self.feature_selection(s, dequants_main, device)
                pose_code = pose_mask*dequants_main
                # ground_truth = pose_mask*dequants_main + (1-pose_mask)*dequants_back
                
                # losses = self.diffusion_fill.training_losses(unet_fill, x_start=pose_code, t=t, x_real_mask=pose_mask) 
                losses = self.diffusion_fill.training_losses(unet_fill, x_start=pose_code, t=t, x_real_mask=pose_mask) 
                loss = losses["loss"] #.mean()
                # print(loss)
                # if torch.isnan(loss):
                #     print(index)
                #     pdb.set_trace()
                t_loss += loss.clone().detach()
                
                index+=1
            
                loss.backward()
                self.optimizer_f.step()

                stats = {
                    'updates': index,
                    'loss': loss.item(),
                    }
                #if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                # updates += 1
            
            if (epoch_i == self.config.epoch) or (epoch_i % 50 == 0):
                print('**********saving ckpt**********')
                checkpoint = {
                        'model': unet_fill.state_dict(),
                        'config': config,
                        'epoch': epoch_i
                    }
                
                ckpt_path = '.experiments/fill_stage/ckpt/'
                if not os.path.exists(ckpt_path):
                        os.makedirs(ckpt_path)
                
                ckpt_path = os.path.join(ckpt_path, str(epoch_i) + '.pth')
                torch.save(checkpoint,ckpt_path)
                    
            print(t_loss/(batch_i+1))
            writer_train.add_scalar('loss_fillzero_fixS', t_loss/(batch_i+1), epoch_i)
            # torch.cuda.empty_cache()
            self.schedular_f.step()   
    
    
    def train_recode(self):
        vqvae = self.modelv.eval()
        unet_fill = self.modelf.eval()
        unet = self.modelu.train()
        
        config = self.config
        data = self.config.data
        training_data = self.training_data 
        testing_data = self.testing_data
        device = torch.device('cuda' if config.cuda else 'cpu')
        # device = torch.device("cuda", self.local_rank)
        log = Logger(self.config, './log/loger/transfer_new/')
        
        # load model
        checkpoint = torch.load(config.vqvae_weight,map_location={'cuda:0':'cuda:1'})
        vqvae.load_state_dict(checkpoint['model'], strict=False)
        
        checkpoint = torch.load(config.fill_weight, map_location={'cuda:0':'cuda:1'})#,
        unet_fill.load_state_dict(checkpoint['model'], strict=False)
                
        if hasattr(config, 'recode_weight') and config.recode_weight is not None and config.recode_weight is not '': 
            checkpoint = torch.load(config.recode_weight, map_location={'cuda:0':'cuda:1'})#,)
            # pdb.set_trace()
            unet.load_state_dict(checkpoint['model'], strict=False)
            
        
        writer_train = SummaryWriter('./log/condition/train/')
        writer_eval = SummaryWriter('./log/condition/eval/')
        
        
        #train model
        for epoch_i in range(1, config.epoch + 1):
            t_loss = 0
            index = 0
            # torch.cuda.empty_cache()
            log.set_progress(epoch_i, len(training_data),self.optimizer_r)
            for batch_i, batch in enumerate(training_data):
                # if batch_i >= 800:
                #     break
                print('########################################')
                # load music and pose data
                music_seq, pose_seq, back_seq = batch
                
                music_seq = music_seq.to(device).to(torch.float)
                music_seq = music_seq.permute(0,2,1)
                # pdb.set_trace()
                pose_seq = pose_seq.to(device)
                back_seq = back_seq.to(device)

                pose_seq[:, :, :3] = 0
                back_seq[:, :, :3] = 0

                self.optimizer_r.zero_grad()
                
                with torch.no_grad():
                    _, dequants_pred_main = vqvae.module.encode(pose_seq)
                    _, dequants_pred_back = vqvae.module.encode(back_seq)
                    
                    if isinstance(dequants_pred_main, tuple):
                        dequants_main = torch.cat([dequants_pred_main[0][0].clone().detach(), dequants_pred_main[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                        dequants_back = torch.cat([dequants_pred_back[0][0].clone().detach(), dequants_pred_back[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                        
                    else:
                        _, dequants_main = vqvae.module.encode(pose_seq)[0][0].clone().detach()
                        _, dequants_back = vqvae.module.encode(back_seq)[0][0].clone().detach()
                
                
                b,c,l = music_seq.shape
                dr=int(pose_seq.shape[1]//dequants_main.shape[2])
                music_seq = music_seq[:,:,:dequants_main.shape[2]*dr] 
                music_seq = music_seq.reshape(b,c*dr,-1) 
                # pdb.set_trace()
                # control the similarity between dancer
                s = random.uniform(0,1)
                pose_mask = self.feature_selection(s, dequants_main, device)
                pose_code = pose_mask*dequants_main
                # ground_truth = pose_mask*dequants_main + (1-pose_mask)*dequants_back
                
                # with torch.no_grad():
                #     # noise = self.diffusion_fill.q_sample(x_start=pose_code, t=torch.full((len(pose_code),),20,device=device))
                #     pose_fill = self.diffusion_fill.ddim_sample_loop(model=unet_fill, shape=pose_code.size(), clip_denoised=False, noise=pose_code, progress=True, device=device, eta=0, timestep=10)
                #     # pdb.set_trace()
                #     pose_f = vqvae.module.quantise_latent(pose_fill)
                    # pdb.set_trace()


                # sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps_r, (len(dequants_main),), device=device).long()
                
                losses = self.diffusion_recode.training_losses(unet, x_start=pose_code, t=t, music=music_seq, seq_main=dequants_main, x_real_mask=pose_mask) 
                
                loss = losses["loss"].mean()
                t_loss += loss.clone().detach()
                
                index+=1
                
                loss.backward()
                self.optimizer_r.step()

                stats = {
                    'updates': index,
                    'loss': loss.item(),
                    }
                #if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                # updates += 1
        
            writer_train.add_scalar('loss_maskinput_usefree', t_loss/(batch_i+1), epoch_i)

            if (epoch_i == self.config.epoch) or (epoch_i % 20 == 0):
                print('**********saving ckpt**********')
                checkpoint = {
                        'model': unet.state_dict(),
                        'config': config,
                        'epoch': epoch_i
                    }
                
                ckpt_path = './experiments/transfer_stage/ckpt/maskinput_usefree/'
                if not os.path.exists(ckpt_path):
                        os.makedirs(ckpt_path)
                
                ckpt_path = os.path.join(ckpt_path, str(epoch_i) + '.pth')
                torch.save(checkpoint,ckpt_path)
            
            # print(t_loss/(batch_i+1))
            if epoch_i % self.config.save_per_epochs == 0:
                t_eval_loss=0
                with torch.no_grad():
                    unet.eval()
                    for batch_i, batch in enumerate(testing_data):
                        # load music and pose data
                        music_seq, pose_seq, back_seq,_ = batch
                        
                        music_seq = music_seq.to(device).to(torch.float)
                        music_seq = music_seq.permute(0,2,1)
                        # pdb.set_trace()
                        pose_seq = pose_seq.to(device)
                        back_seq = back_seq.to(device)

                        pose_seq[:, :, :3] = 0
                        back_seq[:, :, :3] = 0


                        _, dequants_pred_main = vqvae.module.encode(pose_seq)
                        _, dequants_pred_back = vqvae.module.encode(back_seq)
                        
                        if isinstance(dequants_pred_main, tuple):
                            dequants_main = torch.cat([dequants_pred_main[0][0].clone().detach(), dequants_pred_main[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                            dequants_back = torch.cat([dequants_pred_back[0][0].clone().detach(), dequants_pred_back[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                            
                        else:
                            _, dequants_main = vqvae.module.encode(pose_seq)[0][0].clone().detach()
                            _, dequants_back = vqvae.module.encode(back_seq)[0][0].clone().detach()
                        
                        
                        b,c,l = music_seq.shape
                        dr=int(pose_seq.shape[1]//dequants_main.shape[2])
                        music_seq = music_seq[:,:,:dequants_main.shape[2]*dr] 
                        music_seq = music_seq.reshape(b,c*dr,-1) 
                        # pdb.set_trace()
                        # control the similarity between dancer
                        s =  0.5
                        pose_mask = self.feature_selection(s, dequants_main, device)
                        pose_code = pose_mask*dequants_main
                        # ground_truth = pose_mask*dequants_main + (1-pose_mask)*dequants_back
                        
                        # noise = self.diffusion_fill.q_sample(x_start=pose_code, t=torch.full((len(pose_code),),20,device=device))
                        pose_fill = self.diffusion_fill.ddim_sample_loop(model=unet_fill, shape=pose_code.size(), clip_denoised=False, noise=pose_code, progress=True, device=device, eta=0, timestep=10)
                        pose_f = vqvae.module.quantise_latent(pose_fill)

                        # sample t uniformally for every example in the batch
                        t = torch.full((len(pose_code),),200,device=device).long()
                        
                        losses = self.diffusion_recode.training_losses(unet, x_start=pose_f, t=t, music=music_seq, seq_main=dequants_main, seq_back=dequants_back, x_real_mask=pose_mask) 
                        loss = losses["loss"].mean()
                        t_eval_loss += loss

                writer_eval.add_scalar('loss_maskinput_usefree', t_eval_loss/(batch_i+1), epoch_i)
            unet.train()
            self.schedular_r.step()   
                
    def eval(self):
        with torch.no_grad():
            config = self.config
            data = self.config.data
            testing_data = self.testing_data 
            device = torch.device('cuda' if config.cuda else 'cpu')     
            
            vqvae = self.modelv.eval()
            unet_fill = self.modelf.eval()
            unet = self.modelu.eval()
            
            
            # load model
            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)
            
            checkpoint = torch.load(config.fill_weight)
            unet_fill.load_state_dict(checkpoint['model'], strict=False)
            print('**********load unet fill')
            
            checkpoint = torch.load(config.recode_weight )#, map_location={'cuda:0':'cuda:1'})
            unet.load_state_dict(checkpoint['model'], strict=False)
            print('**********load unet recode')
            
            sl=[]
            stt=[]
            gsl=[]
            att=[]

            for batch_i, batch in enumerate(testing_data):
                if '302' not in self.test_name[batch_i]:
                    continue

                print('########################################')
                # if batch_i < 800:
                #     continue
                # load music and pose data
                music_seq, pose_seq, back_seq, name = batch
                
                music_seq = music_seq.to(device).to(torch.float)
                music_seq = music_seq.permute(0,2,1)
                
                pose_seq = pose_seq.to(device)
                back_seq = back_seq.to(device)
                
                pose_seq[:, :, :3] = 0
                back_seq[:, :, :3] = 0
                
                _, dequants_pred_main = vqvae.module.encode(pose_seq)
                _, dequants_pred_back = vqvae.module.encode(back_seq)
                
                if isinstance(dequants_pred_main, tuple):
                    dequants_main = torch.cat([dequants_pred_main[0][0].clone().detach(), dequants_pred_main[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                    dequants_back = torch.cat([dequants_pred_back[0][0].clone().detach(), dequants_pred_back[1][0].clone().detach()],axis=2) #(batchsize,512,seqlength/8)
                    
                    dequants_main_last_frame_up = dequants_main[:, :, dequants_main.shape[-1]//2 -1].unsqueeze(2)
                    dequants_main_last_frame_down = dequants_main[:, :, -1].unsqueeze(2)
                    dequants_back_last_frame_up = dequants_back[:, :, dequants_main.shape[-1]//2 -1].unsqueeze(2)
                    dequants_back_last_frame_down = dequants_back[:, :, -1].unsqueeze(2)
                    
                    # append the last frame if cant be divided by 64 until can be divided by 64
                    if dequants_main.size(2) % 64 != 0:
                        repeat_len = (64 - dequants_back.size(2) % 64) // 2
                        B, C, T = dequants_main.shape
                        
                        dequants_main = torch.cat([dequants_main[:,:,:T//2], dequants_main_last_frame_up.repeat(1, 1, repeat_len),dequants_main[:,:,T//2:], dequants_main_last_frame_down.repeat(1, 1, repeat_len)], axis=2)
                        dequants_back = torch.cat([dequants_back[:,:,:T//2], dequants_back_last_frame_up.repeat(1, 1, repeat_len),dequants_back[:,:,T//2:], dequants_back_last_frame_down.repeat(1, 1, repeat_len)], axis=2)
                        # breakpoint()
                        # [1, 419, 1792]
                        # pdb.set_trace()
                        if music_seq.shape[2] < dequants_main.shape[2]*4:
                            music_seq = torch.cat([music_seq, torch.zeros(music_seq.size(0),music_seq.size(1),(dequants_main.shape[2]*4-music_seq.shape[2])).to(device)],axis=2)
                        else:
                            music_seq = music_seq[:,:,:dequants_main.shape[2]*4]
                        # breakpoint()
                        # reshape to [1, 419*4, dequants_main.shape[2]]
                        # [1, 1676, 448]
                        music_seq = music_seq.reshape(music_seq.size(0), music_seq.size(1)*4, -1)
                    
                    bs = int(dequants_main.size(2) // 64) 
                    dequants_main_re = []
                    dequants_back_re = []
                    
                    for b in range(bs):
                        dequants_main_re.append(torch.cat([dequants_main[:,:,b*32:(b+1)*32],dequants_main[:,:,T//2+b*32:T//2+(b+1)*32]],axis=2))
                        dequants_back_re.append(torch.cat([dequants_back[:,:,b*32:(b+1)*32],dequants_back[:,:,T//2+b*32:T//2+(b+1)*32]],axis=2))

                    # dequants_main = torch.cat(torch.split(dequants_main,bs,dim=-1),dim=0) ##[bs,512,448/64]
                    # dequants_back = torch.cat(torch.split(dequants_back,bs,dim=-1),dim=0)
                    dequants_main_re = torch.stack(dequants_main_re).reshape(-1,C,64)
                    dequants_back_re = torch.stack(dequants_back_re).reshape(-1,C,64)
                    music_seq_re = torch.cat(torch.split(music_seq,music_seq.shape[-1]//bs,dim=-1),dim=0) ##[bs,419*4,448/64]
                else:
                    _, dequants_main = vqvae.module.encode(pose_seq)[0][0].clone().detach()
                    _, dequants_back = vqvae.module.encode(back_seq)[0][0].clone().detach()
                


                # control the similarity between dancer
                # s = random.uniform(0,1)
                # for s in range(0,10,2):
                # # s = 0.5  #similarity control
                #     pose_mask = self.feature_selection(s*0.1, dequants_main, device)  #(bs,512,len)
                #     pose_code = pose_mask*dequants_main
                #     ground_truth = pose_mask*dequants_main + (1-pose_mask)*dequants_back
                #     # print(torch.min(ground_truth),torch.max(ground_truth))
                #     # pdb.set_trace()
                #     # idx = torch.nonzero(pose_mask[-1,-1]).reshape(-1)
                #     # sim=idx.shape[0]
                #     # sim=torch.tensor(sim).to(pose_mask.device)
                #     # sim=sim.repeat(pose_mask.shape[0])
                #     noise = self.diffusion_fill.q_sample(x_start=pose_code, t=torch.full((len(pose_code),),10,device=device))
                #     result = self.diffusion_fill.ddim_sample_loop(model=unet_fill, shape=pose_code.size(), clip_denoised=False, noise=noise, progress=True, device=device, eta=0.5, timestep=10)
                    
                #     _, _, t = result.size()
                #     pose_decode, pose_c = vqvae.module.decode_latent(([result[:, :, :t//2]],[result[:, :, t//2:]]))  #(up,down) #(batch,512,len)
                #     real_decode, real_c = vqvae.module.decode_latent(([ground_truth[:, :, :t//2]],[ground_truth[:, :, t//2:]]))
                #     print(torch.nonzero(pose_c==real_c).shape)
                #     print(torch.nonzero(pose_mask[0][0]).shape)
                # pdb.set_trace()
                #tata=[0,2,4,6,8,10]
                    
                sa = [2,6,10]
                for guidance_scale in sa:
                    ata = 0.9
                    s=0.2
                    print('similar',s)
                    pose_mask = self.feature_selection(s, dequants_main_re, device)  #(bs,512,len)
                    pose_code = pose_mask*dequants_main_re
                    ground_truth = pose_mask*dequants_main_re + (1-pose_mask)*dequants_back_re

                    c=0
                    ss=0
                    gs=0
                    at=0
                    # for guidance_scale in range(2,10,2):
                        
                    # guidance_scale=9
                    # noise = self.diffusion_fill.q_sample(x_start=pose_code, t=torch.full((len(pose_code),),20,device=device))
                    # result = self.diffusion_fill.ddim_sample_loop(model=unet_fill, shape=pose_code.size(), clip_denoised=False, noise=pose_code, progress=True, device=device, eta=i/2, timestep=50)
                    pose_fill = self.diffusion_fill.ddim_sample_loop(model=unet_fill, shape=pose_code.size(), clip_denoised=False, noise=pose_code, progress=True, device=device, eta=0, timestep=10)
                    pose_f = vqvae.module.quantise_latent(pose_fill)

                    # # for ata in range(1,10,1):
                    # ata=9
                    noise = self.diffusion_recode.q_sample(x_start=pose_f, t=torch.full((len(dequants_main_re),),50,device=device))
                    # # result = self.diffusion_recode.classifier_free_sample_loop_progressive(model=unet, clip_denoised=False, shape=pose_code.size(), noise=noise, progress=True, device=device, eta=0, timestep=100,  do_classifier_free=True,pose_mask=pose_mask)
                    result = self.diffusion_recode.classifier_free_sample_loop_progressive(model=unet, clip_denoised=False, shape=pose_code.size(), noise=noise, progress=True,
                    device=device, eta=0, timestep=50, music=music_seq_re, seq_main=dequants_main_re, do_classifier_free=True, pose_mask=pose_mask, guidance_scale=guidance_scale,ata=ata*0.1)
                        
                    result = [sample for j, sample in enumerate(result)]
                    
                    # # for step in range(0,len(result)+1,10):
                    step=50
                    # print(step)
                    fin = result[step-1]['sample']
                    # fin=pose_f
                    T=fin.shape[-1]
                    final_up = fin[0][:,:T//2]
                    final_down = fin[0][:,T//2:]
                    gt_up = ground_truth[0][:,:T//2]
                    gt_down = ground_truth[0][:,T//2:]
                    # main_up = dequants_main_re[0][:,:T//2]
                    # main_down = dequants_main_re[0][:,T//2:]
                    for f in range(1,len(fin)):
                        final_up = torch.cat([final_up,fin[f][:,:T//2]],axis=-1)
                        final_down = torch.cat([final_down,fin[f][:,T//2:]],axis=-1)
                        gt_up = torch.cat([gt_up,ground_truth[f][:,:T//2]],axis=-1)
                        gt_down = torch.cat([gt_down,ground_truth[f][:,T//2:]],axis=-1)
                        # main_up = torch.cat([main_up,dequants_main_re[f][:,:T//2]],axis=-1)
                        # main_down = torch.cat([main_down,dequants_main_re[f][:,T//2:]],axis=-1)
                    
                    final_up=final_up.reshape(1,final_up.shape[0],final_up.shape[1])
                    final_down=final_down.reshape(1,final_down.shape[0],final_down.shape[1])
                    gt_up=gt_up.reshape(1,gt_up.shape[0],gt_up.shape[1])
                    gt_down=gt_down.reshape(1,gt_down.shape[0],gt_down.shape[1])
                    # main_up=main_up.reshape(1,main_up.shape[0],main_up.shape[1])
                    # main_down=main_down.reshape(1,main_down.shape[0],main_down.shape[1])

                    # _, _, t = gt.size()
                    # pose_decode, pose_c = vqvae.module.decode_latent(([final[:, :, :t//2]],[final[:, :, t//2:]]))  #(up,down) #(batch,512,len)
                    # real_decode, real_c = vqvae.module.decode_latent(([gt[:, :, :t//2]],[gt[:, :, t//2:]]))
                    # main_decode, main_c = vqvae.module.decode_latent(([main[:, :, :t//2]],[main[:, :, t//2:]]))
                    pose_decode, pose_c = vqvae.module.decode_latent(([final_up],[final_down])) 
                    real_decode, real_c = vqvae.module.decode_latent(([gt_up],[gt_down]))
                    # main_decode, main_c = vqvae.module.decode_latent(([main_up],[main_down]))
                    main_decode, main_c = vqvae.module.decode_latent(([dequants_main[:,:,:dequants_main.shape[-1]//2]],[dequants_main[:,:,dequants_main.shape[-1]//2:]]))
                    print(pose_c)
                    sim = torch.nonzero(pose_c[0]==real_c[0]).shape[0]/real_c.shape[-1]
                    print(sim)
                    pdb.set_trace()
                    # if sim >c:
                    #     c=sim
                    #     ss=step
                    #     gs=guidance_scale
                    #     at=ata

                        #     sl.append(c)
                        #     stt.append(ss)
                        #     gsl.append(gs)
                        #     att.append(at)
                        #     print(sl)
                    # pdb.set_trace()

                            
                    # pdb.set_trace()
                        # result = unet(noise,torch.full((len(dequants_main),),19,device=device), m=music_seq, s1=pose_mask*dequants_main)
                        
                        # print(torch.min(result),torch.max(result))
                        # pdb.set_trace()

                        # result = unet(noise,
                        # torch.full((len(dequants_main),),20,device=device),
                        # m=music_seq, 
                        # s1=pose_mask*dequants_main,
                        # s2=(1-pose_mask)*dequants_back)

                        # _, _, t = result.size()
                        # pose_decode, pose_c = vqvae.module.decode_latent(([result[:, :, :t//2]],[result[:, :, t//2:]]))
                        # print(torch.nonzero(pose_c==real_c).shape)
                        # pdb.set_trace()

                        
                    # noise = self.diffusion_recode.q_sample(x_start=result, t=torch.full((len(dequants_main),),50,device=device))
                    # model_output = unet(noise, torch.full((len(dequants_main),),50,device=device), music_seq, dequants_main, dequants_back,)
                    # pose_decode, pose_c = vqvae.module.decode_latent(([model_output[:, :, :t//2]],[model_output[:, :, t//2:]]))
                    # pdb.set_trace()
                    
                    # b, t, c = pose_decode.shape
                    # pose_decode = pose_decode.reshape(b,t,c//3,3)
                    # real_decode = real_decode.reshape(b,t,c//3,3)
                    # mse = torch.nn.MSELoss()
                    # tot_loss = mse(pose_decode, real_decode)
                    pose_decode = pose_decode.cpu().data.numpy()[0]
                    real_decode = real_decode.cpu().data.numpy()[0]
                    main_decode = main_decode.cpu().data.numpy()[0]

                    root = pose_decode[:, :3]
                    pose_decode = pose_decode + np.tile(root, (1, 24))
                    pose_decode[:, :3] = root
                    
                    root = real_decode[:, :3]
                    real_decode = real_decode + np.tile(root, (1, 24))
                    real_decode[:, :3] = root
                    
                    root = main_decode[:, :3]
                    main_decode = main_decode + np.tile(root, (1, 24))
                    main_decode[:, :3] = root
                    
                    epoch_i=200
                    st = 50
                    ep_path = os.path.join('./exp_ckpt', "eval/final/s0.2_gs"+str(guidance_scale)+"_final/", f"ep{epoch_i:06d}", f"NoiseStep{st:d}")
                    if not os.path.exists(ep_path):
                        os.makedirs(ep_path)
                    
                    print('**********Generating Jsons**********')
                    
                    # pkl_data = {'pred_latent': result, "pred_position": pose_decode, "pred_code":pose_c, 'real_latent': ground_truth, "real_position": real_decode, "real_code":real_c, "pose_fill":pose_fill}
                    
                    # pkl_data = {'pred_code':pose_c,'real_code':real_c,'mask':pose_mask,'pose_decode':pose_decode,'real_decode':real_decode,'similar':s}
                    # dance_path = os.path.join(ep_path, str(name) )
                    
                    # np.save(dance_path, pkl_data)
                    pkl_data = {'leader_name':name[0][:-4],'pred_code':pose_c,'real_code':real_c,'mask':pose_mask,'pose_decode':pose_decode,'main_decode':main_decode,'real_decode':real_decode,'similar':s, 'count':torch.nonzero(pose_c[0]==real_c[0]).shape[0] }
                    dance_path = os.path.join(ep_path, str(self.test_name[batch_i]) )
                    
                    np.save(dance_path, pkl_data)

        
            
                
