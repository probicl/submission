**Installation**

Setup [miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
```
(Install miniconda3 using the above script)
conda activate
conda create -n my_env
conda activate my_env
conda install -c conda-forge mamba libarchive
mamba update --all
mamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

(Install following packages or equivalent)
apt install freeglut3-dev libosmesa6-dev patchelf xvfb
```

[Mujoco](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0)
```
tar xf MUJOCO.tar.gz -o
pip install mujoco_py
(Ensure you can import mujoco_py in python correctly)
```

Install tools
```
pip install "cython<3"
pip install "gym<0.26.0"
pip install .
pip install torch --upgrade
pip install pynvml
```

**Running code**

Comment/uncomment code before running
```
ENV="gridworld" # one of ['gridworld', 'cartpole', 'mujoco_ant', 'mujoco_hc', 'highd']
SEED=0
BETA=0.99
DELTA=0.9
OUTPUT_DIR=output
python3 -B icl.py -env "$ENV" -o "$OUTPUT_DIR" -seed 0 -beta "$BETA"
python3 -B prob_icl.py -env "$ENV" -o "$OUTPUT_DIR" -seed 0 -beta "$BETA" -delta "$DELTA"
```
