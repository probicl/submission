import torch
import glob
import os

CLEAR_FILES = False
device = torch.device("cpu")
# torch.cuda.set_per_process_memory_fraction(0.19)
env_name = None
env_names_allowed = ['gridworld', 'cartpole']
continuous_actions = False
output_dir = None
expert_data_file = None
seed = 0
n_actions = None
constraint_fn_input_dim = None
hidden_dim = 64
discount_factor = None
episodes_per_epoch = 20
ppo_subepochs = 25
replay_buffer_size = 10000
learning_rate = 5e-4
learning_rate_feasibility = 2.5e-5
minibatch_size = 64
clip_param = 0.1
entropy_coef = 0.01
beta = None
alpha = 15
n_iters = 10
flow_iters = 20
ppo_iters = None
policy_add_to_mix_every = None
ca_iters = 20
n_trajs = 50
delta = None
sigmoid_lambda = 100.
small_eps = 0.1
ppo_sigmoid = False # use sigmoid approximation in PPO penalty
