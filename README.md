# Dance with You: The Diversity Controllable Dancer Generation via Diffusion Models
Official Code for ACM MM 2023 paper "Dance with You: The Diversity Controllable Dancer Generation via Diffusion Models"

> Recently, digital humans for interpersonal interaction in virtual environments have gained significant attention. In this paper, we introduce a novel multi-dancer synthesis task called partner dancer generation, which involves synthesizing virtual human dancers capable of performing dance with users. The task aims to control the pose diversity between the lead dancer and the partner dancer. The core of this task is to ensure the controllable diversity of the generated partner dancer while maintaining temporal coordination with the lead dancer. This scenario varies from earlier research in generating dance motions driven by music, as our emphasis is on automatically designing partner dancer postures according to pre-defined diversity, the pose of lead dancer, as well as the accompanying tunes. To achieve this objective, we propose a three-stage framework called \textbf{Dan}ce-with-\textbf{Y}ou (\textbf{DanY}). Initially, we employ a 3D Pose Collection stage to collect a wide range of basic dance poses as references for motion generation. Then, we introduce a hyper-parameter that coordinates the similarity between dancers by masking poses to prevent the generation of sequences that are over-diverse or consistent. To avoid the rigidity of movements, we design a Dance Pre-generated stage to pre-generate these masked poses instead of filling them with zeros. After that, a Dance Motion Transfer stage is adopted with leader sequences and music, in which a multi-conditional sampling formula is rewritten to transfer the pre-generated poses into a sequence with a partner style. In practice, to address the lack of multi-person datasets, we introduce AIST-M, a new dataset for partner dancer generation. Comprehensive evaluations on our AIST-M dataset demonstrate that the proposed DanY can synthesize satisfactory partner dancer results with controllable diversity.

<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/JJessicaYao/AIST-M-Dataset.git
   ```
3. Install python packages
   ```sh
   PyTorch == 1.6.0
   ```

### AIST-M-Dataset

This is the dataset proposed in **Dance with You: The Diversity Controllable Dancer Generation via Diffusion Models**. You can download our dataset at [here](https://drive.google.com/uc?export=download&id=1--4lDL71kkbo-uPm5ZbqpISKTu7CgCH1).

#### Data Structure
The data directory is organized as follows:
* Keypoints2D: The 2D COCO-format keypoints with the shape (num_views x num_frames x 17 x 3).
* Keypoints3D: The 3D COCO-format keypoints with the shape (num_frames x 17 x 3).
* Motions: The SMPL pose format keypoints with 'smpl_poses', 'root_trans' and 'scaling'. 
  > 'smpl_poses': shape (num_frames x 72): the motions contain 72-D vector pose parameters in SMPL pose format.
  > 
  > 'root_trans': shape (num_frames x 3): sequences of root translation.
  > 
  > 'scaling': shape (1 x 1): scaling factor of human body size.

* Lead-Partner: The Lead-Partner dancer pairs composed of SMPL joints.
  > 'leader_name': the sequence name of the lead dancer.
  > 
  > 'lead_dancer': the motion sequence of the lead dancer with shape (num_frames x 72).
  > 
  > 'partner_dancer': the motion sequence of the corresponding partner dancer (num_frames x 72).
* music_feature: The features are extracted from raw music file. The music frames are aligned with the motion frames.

#### Usage
In our experiments, please unzip the zip file into './dataset/' folder before training.  

<!-- Training -->
## Training
The training of DanY comprises of 3 steps in the following sequence. 
### Step 1: Training the 3D Pose Collection Stage
The implementation of this step requires all of the dance motion data, both single and multiple dancers. You can choose to download the dataset of single dancer from [AIST++](https://google.github.io/aistplusplus_dataset/download.html) and mix its motion files with our AIST-M motion files to train from scratch. Alternatively, you can directly download pre-trained checkpoints for single dance from [Bailando's](https://github.com/lisiyao21/Bailando/tree/main) GitHub and fine-tune them using the motion files from our AIST-M dataset.
```sh
sh srun_vqvae.sh configs/sep_vqvae.yaml train [your node name] 1
```
### Step 2: Training the Dance Pre-generation Stage
The implementation of this step requires the 'lead_dancer' part in './dataset/lead_partner/' folder. 
```sh
sh srun.sh configs/diff_vqvae.yaml train [your node name] 1
```
### Step 3: Training the Dance Motion Transfer Stage
The implementation of this step requires the 'partner_dancer' part in './dataset/lead_partner/' folder and the results generated in step 2. 
```sh
sh srun.sh configs/diff_vqvae.yaml train [your node name] 1
```

<!-- Evaluation -->
## Evaluation
### For qualitative evaluation 
The training results of the previous steps will be stored in the './exp_ckpt/' folder.
```sh
sh srun.sh configs/diff_vqvae.yaml eval [your node name] 1
```

### For quantitative evaluation 
After generating the dance in the above step, run the following codes.
```sh
python evaluate.py
```

## Citation

    @inproceedings{yao2023dance,
	    title={Dance with you: The diversity controllable dancer generation via diffusion models},
        author={Yao, Siyue and Sun, Mingjie and Li, Bingliang and Yang, Fengyu and Wang, Junle and Zhang, Ruimao},
        booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
        pages={8504--8514},
        year={2023}
    }

    
