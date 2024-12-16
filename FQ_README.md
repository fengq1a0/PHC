# Data format
merge into original pkl
img = [L, 1024]
bbox = [L, 3] cx, cy, b
kp2d = [L, 34]
camera: 
R    = [L, 9]
vfov = [L, 1]

10+1024+37 = 1071

# Set up the environment


## python 3.8   cuda 12.1
conda create -n isaac python=3.8
conda activate isaac
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirement.txt
> version: rl-games==1.1.4

## isaac gym
> get the env in your dic to avoid other issues
tar zxvf isaacgym to this directory
cd isaacgym/python
pip install -e .

## Download SMPL and SMPLX files
npz files for SMPLX are also needed.


## Download Data
bash download_data.sh


## add this to all your bash scripts
export LD_LIBRARY_PATH="/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/envs/gym/lib:$LD_LIBRARY_PATH"

## Line: 135 in PHC/isaacgym/python/isaacgym/torch_utils.py
np.float --> np.float32

## For chumpy and numpy conflict
in __init__.py
#from numpy import bool, int, float, complex, object, unicode, str, nan, inf



# data
data/amass_smpl/train.pkl
data/behave/*