''' The code was forked from https://github.com/NVlabs/SPADE. '''

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import sync_batchnorm 
import torch.nn.utils.spectral_norm as spectral_norm



# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, param_free_norm_type, norm_nc, label_nc, kernel_size=3, dim=1):
        super().__init__()

        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        # ks = int(parsed.group(2))
        
        if dim == 1:
            if param_free_norm_type == 'instance':
                self.param_free_norm = nn.InstanceNorm1d(norm_nc, affine=False)
            elif param_free_norm_type == 'syncbatch':
                self.param_free_norm = sync_batchnorm.SynchronizedBatchNorm1d(norm_nc, affine=False)
            elif param_free_norm_type == 'batch':
                self.param_free_norm = nn.BatchNorm1d(norm_nc, affine=False)
            else:
                raise ValueError('%s is not a recognized param-free norm type in SPADE'
                                % param_free_norm_type)
                
            # The dimension of the intermediate embedding space. Yes, hardcoded.
            nhidden = 512

            pw = kernel_size // 2
            
            self.mlp_shared = nn.Sequential(
                nn.Conv1d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv1d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
            self.mlp_beta = nn.Conv1d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
            
        elif dim == 2:
            if param_free_norm_type == 'instance':
                self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
            elif param_free_norm_type == 'syncbatch':
                self.param_free_norm = sync_batchnorm.SynchronizedBatchNorm2d(norm_nc, affine=False)
            elif param_free_norm_type == 'batch':
                self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
            else:
                raise ValueError('%s is not a recognized param-free norm type in SPADE'
                                % param_free_norm_type)

            # The dimension of the intermediate embedding space. Yes, hardcoded.
            nhidden = 128

            pw = kernel_size // 2
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
            
        elif dim == 3:
            if param_free_norm_type == 'instance':
                self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
            elif param_free_norm_type == 'syncbatch':
                self.param_free_norm = sync_batchnorm.SynchronizedBatchNorm3d(norm_nc, affine=False)
            elif param_free_norm_type == 'batch':
                self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
            else:
                raise ValueError('%s is not a recognized param-free norm type in SPADE'
                                % param_free_norm_type)

            # The dimension of the intermediate embedding space. Yes, hardcoded.
            nhidden = 128

            pw = kernel_size // 2
            self.mlp_shared = nn.Sequential(
                nn.Conv3d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
            self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
            
        else:
            raise ValueError('%s is out of range in SPADE'
                                % dim)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * norm_nc if use_scale_shift_norm else norm_nc,
            ),
        )
        
    def forward(self, x, segmap, use_emb=False):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        
        if use_emb:
            emb_out = self.emb_layers(emb).type(normalized.dtype)
            out = normalized * (1 + gamma) + beta + emb

        return out
    
    
# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv1d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv1d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv1d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        # spade_config_str = opt.norm_G.replace('spectral', '')
        spade_config_str = 'instance'
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

