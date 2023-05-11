# AIST-M-Dataset

This is the dataset proposed in **Dance with You: The Diversity Controllable Dancer Generation via Diffusion Models**.

# Data Structure
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
