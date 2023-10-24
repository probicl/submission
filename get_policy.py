import basic
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import tools
tools.utils.nowarnings()
import utils
import torch
import matplotlib.pyplot as plt
import tqdm
import atexit
import argparse
import glob
import datetime

# v1 => PPO-Pen with sigmoid + CA with x/sqrt(x^2+eps)
# v2 => PPO-Pen with x/sqrt(x^2+eps) + CA with x/sqrt(x^2+eps)

parser = argparse.ArgumentParser()
parser.add_argument("-o", type=str, required=False, default="demo", help="Output directory (overwrite)")
parser.add_argument("-seed", type=int, required=False, default=0, help="Training seed")
parser.add_argument("-env", type=str, required=True, help="Environment name")
parser.add_argument("-beta", type=float, required=False, default=-1, help="Prob ICL beta")
parser.add_argument("-delta", type=float, required=False, default=0.1, help="Prob ICL delta")
parser.add_argument("-type", type=int, required=False, default=2, help="Approximation type")
args = parser.parse_args()
# torch.cuda.set_device(args.seed)

basic.true_constraint_function = utils.true_constraint_function
basic.env_name = args.env
basic.output_dir = args.o
basic.expert_data_file = "%s/expert_data.pt" % basic.output_dir
if not os.path.exists(basic.output_dir):
    os.mkdir(basic.output_dir)
if basic.CLEAR_FILES:
    for f in glob.glob('%s/*' % basic.output_dir):
        os.remove(f)
basic.seed = args.seed
basic.delta = args.delta
print("Delta: %g" % (basic.delta))
if args.type == 1:
    basic.ppo_sigmoid = True
elif args.type == 2:
    basic.ppo_sigmoid = False

atexit.register(utils.end_logging)
utils.start_logging("%s/0_log.txt" % basic.output_dir)
print("Accrual ... %s" % (datetime.datetime.utcnow()))
utils.seed_fn(basic.seed)

basic.env = utils.create_env()
basic.obs_n = basic.env.observation_space.shape[0]
basic.act_n = basic.env.action_space.n
basic.env.seed(basic.seed)

if args.beta != -1:
    basic.beta = args.beta
print("Beta: %g" % (basic.beta))

saved_chkpts = glob.glob("%s/iter_*.pt" % basic.output_dir)

assert(len(saved_chkpts) > 0)

iter_numbers = [int(item.split("_")[-1].split(".")[0]) for item in saved_chkpts]
max_iter_number = max(iter_numbers)
restore_file = "%s/iter_%d.pt" % (basic.output_dir, max_iter_number)
print("Restoring to ... %s" % restore_file)
basic.constraint_nn, basic.constraint_opt, \
    basic.policy_mixture, basic.weights_mixture = torch.load(restore_file)
basic.constraint_nn = basic.constraint_nn.to(basic.device)
basic.expert_data = utils.generate_expert_data(basic.env)
basic.agent_data = None

print("Final policy learning ... %s" % (datetime.datetime.utcnow()))
value_nn, policy_nn = utils.make_nn()
utils.ppo_penalty(basic.ppo_iters, basic.env, policy_nn, value_nn, utils.current_constraint_function, p=True)
basic.agent_data = utils.collect_trajectories(\
    len(basic.expert_data),
    basic.env, 
    policy_nn, 
    utils.current_constraint_function
)
torch.save([basic.agent_data, value_nn, policy_nn], "%s/final_policy.pt" % (basic.output_dir))

print("Finished ... %s" % (datetime.datetime.utcnow()))
