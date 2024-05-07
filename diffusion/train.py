"""
train model only use diffusion and mask
"""

import functools
import numpy as np
import torch
import pdb

from diffusion_model import get_named_beta_schedule, GaussianDiffusion
from unet import UNetModel
from torch.optim import AdamW
from PIL import Image
import requests
from torchvision import transforms
from torch.utils.data import DataLoader

# load dataset from the hub
import mnist_reader
dataset = mnist_reader.load_mnist('./fashion-mnist/data/fashion', kind='train')
#dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128

# define image transformations (e.g. using torchvision)
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

# define function
def transforms(examples):
   [examples]["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

print(dataset[0].shape)
transformed_dataset = (dataset[0].reshape(-1,1,image_size,image_size)/255)*2-1
re_dataset=np.zeros((transformed_dataset.shape[0],1,32,32))
re_dataset[:,:,2:-2,2:-2]=transformed_dataset

# create dataloader
dataloader = DataLoader(re_dataset, batch_size=batch_size, shuffle=True)

timesteps=200
beats=get_named_beta_schedule('linear',timesteps)
diffmodel=GaussianDiffusion(betas=beats,model_mean_type='xstart',model_var_type='learned',loss_type='mse')

device="cuda" if torch.cuda.is_available() else "cpu"


image_size=32

attention_ds = []
attention_resolutions="16"
for res in attention_resolutions.split(","):
    attention_ds.append(image_size // int(res))
    
model=UNetModel(
    image_size=image_size,
    in_channels=1,
    model_channels=128,
    out_channels=2,
    num_res_blocks=2,
    attention_resolutions=tuple(attention_ds),
    dropout=0,
    channel_mult=(1, 1, 2, 3, 4),
    num_classes=None,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_new_attention_order=False,
)

model.to(device)
optimizer=AdamW(model.parameters(),lr=1e-4)


from torchvision.utils import save_image

epochs = 50

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

from pathlib import Path
results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        batch_size = batch.shape[0]
        batch = batch.to(device)

        batch=batch.type(torch.float32)
        # print('batch',batch.dtype)
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        # print('t',t.shape)
        # print('batch',batch.shape)
        
        loss = diffmodel.training_losses(model, batch, t, batch )
        pdb.set_trace()
        if step % 100 == 0:
            print("Loss:", np.sum(loss.tolist()))

        loss.backward(torch.ones_like(loss))
        optimizer.step()

        # save generated images
        # if step != 0 and step % 100 == 0:
        #     milestone = step // 100
        #     batches = num_to_groups(4, batch_size)
        #     print(batches)
        #     all_images_list = list(map(lambda n: diffmodel.p_sample_loop(denoise_fn=model, shape=[batch_size,1,image_size,image_size], device=device), batches))
        #     all_images = torch.cat(all_images_list, dim=0)
        #     all_images = (all_images + 1) * 0.5
        #     save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
        
        if np.sum(loss.tolist()) < 10 and step % 100 == 0:
            milestone = step // 100
            batches = num_to_groups(4, batch_size)
            print(batches)
            all_images_list = list(map(lambda n: diffmodel.p_sample_loop(denoise_fn=model, shape=[batch_size,1,image_size,image_size], device=device), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
        
