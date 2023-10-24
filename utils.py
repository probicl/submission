import basic
import copy
import logging
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import tools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from scipy.spatial.distance import cdist
import ot
import gc

# https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            try:
                f.flush()
            except:
                pass

def start_logging(filename):
    if not os.path.exists(filename):
        basic.log_file = open(filename, 'w')
    else:
        basic.log_file = open(filename, 'a')
    sys.stdout = Tee(sys.stdout, basic.log_file)
    print(" ".join("\""+arg+"\"" if " " in arg else arg for arg in sys.argv))

def end_logging():
    basic.log_file.close()

def seed_fn(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.memory._record_memory_history(enabled=True)

def true_constraint_function(sa):
    if basic.env_name == 'gridworld':
        s, a = sa
        x, y = s[0], s[1]
        u = [(ui, uj) for ui in [3] for uj in [0, 1, 2, 3]]
        if (x, y) in u:
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()
    elif basic.env_name == 'cartpole':
        s, a = sa
        # if s[0] <= 0.2:
        # if (s[0] > 1. and a == 1) or (s[0] < -1. and a == 0):
        # if (s[0] <= -1.75 and a == 0) or (-1 <= s[0] <= 1) or (s[0] >= 1.75 and a == 1):
        if (s[0] < -1 or s[0] > 1):
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()
    else:
        print('Bad env_name')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

def constraint_mse(true_constraint_fn, constraint_fn):
    if true_constraint_fn == None or constraint_fn == None:
        return None
    if basic.env_name == 'gridworld':
        grid_for_action, true_grid_for_action = [], []
        for a in np.arange(basic.n_actions):
            grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
            true_grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
            for x in np.arange(basic.gridworld_dim):
                for y in np.arange(basic.gridworld_dim):
                    grid[x, y] = constraint_fn(([x, y], a)).item()
                    true_grid[x, y] = true_constraint_fn(([x, y], a)).item()
            grid_for_action += [grid]
            true_grid_for_action += [true_grid]
        average_grid = np.mean(grid_for_action, axis=0)
        true_average_grid = np.mean(true_grid_for_action, axis=0)
        return mse(true_average_grid, average_grid)
    elif basic.env_name == 'cartpole':
        a0, a1 = [], []
        true_a0, true_a1 = [], []
        for x in np.arange(-2.4, 2.4+0.1, 0.1):
            a0 += [constraint_fn(([x], 0)).item()]
            a1 += [constraint_fn(([x], 1)).item()]
            true_a0 += [true_constraint_fn(([x], 0)).item()]
            true_a1 += [true_constraint_fn(([x], 1)).item()]
        true_vals = np.array([true_a0, true_a1])
        vals = np.array([a0, a1])
        return mse(true_vals, vals)
    else:
        print('Bad env_name')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

def accrual_comparison(expert_data, data):
    if expert_data == None or data == None:
        return None
    if basic.env_name == 'gridworld':
        accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in data:
            for s, a in zip(S, A):
                x, y = s
                accrual[a][x][y] += 1
        accrual = np.mean(accrual, axis=0)
        accrual /= (np.max(accrual)+1e-6)
        expert_accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in expert_data:
            for s, a in zip(S, A):
                x, y = s
                expert_accrual[a][x][y] += 1
        expert_accrual = np.mean(expert_accrual, axis=0)
        expert_accrual /= (np.max(expert_accrual)+1e-6)
        return wasserstein_distance2d(expert_accrual, accrual)
    elif basic.env_name == 'cartpole':
        rng = np.arange(-2.4, 2.4+0.1, 0.1)
        accrual = np.zeros((2, len(rng)))
        for S, A in data:
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 49)
                accrual[int(a)][bin_nbr] += 1
        accrual[0, :] /= (accrual[0, :].max()+1e-6)
        accrual[1, :] /= (accrual[1, :].max()+1e-6)
        expert_accrual = np.zeros((2, len(rng)))
        for S, A in expert_data:
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 49)
                expert_accrual[int(a)][bin_nbr] += 1
        expert_accrual[0, :] /= (expert_accrual[0, :].max()+1e-6)
        expert_accrual[1, :] /= (expert_accrual[1, :].max()+1e-6)
        return 0.5*(wasserstein_distance2d(expert_accrual[0, :].reshape(1, -1), accrual[0, :].reshape(1, -1))+\
            wasserstein_distance2d(expert_accrual[1, :].reshape(1, -1), accrual[1, :].reshape(1, -1)))
    else:
        print('Bad env_name')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

def show_metrics(p=False):
    expert_satisfaction = \
        compute_current_constraint_value_trajectory(\
            basic.constraint_nn, 
            basic.expert_data,
            p=p\
        ).mean()
    if p:
        print("Expert => P_aprx(C<=beta) = %.2f, beta = %.2f, delta = %.2f" % (expert_satisfaction, basic.beta, basic.delta))
        expert_cdf = compute_cdf(basic.constraint_nn, basic.expert_data)
        agent_cdf = compute_cdf(basic.constraint_nn, basic.agent_data)
        if expert_cdf != None:
            print("Expert => P_empr(C<=beta) = %.2f, beta = %.2f, delta = %.2f" % (expert_cdf, basic.beta, basic.delta))
        else:
            print("Expert => P_empr(C<=beta) = None, beta = %.2f, delta = %.2f" % (basic.beta, basic.delta))
        if agent_cdf != None:
            print("Agent => P_empr(C<=beta) = %.2f, beta = %.2f, delta = %.2f" % (agent_cdf, basic.beta, basic.delta))
        else:
            print("Agent => P_empr(C<=beta) = None, beta = %.2f, delta = %.2f" % (basic.beta, basic.delta))
    else:
        print("Expert => ExpSat = %.2f, beta = %.2f" % (expert_satisfaction, basic.beta))
        expert_satisfaction2 = \
        compute_current_constraint_value_trajectory(\
            basic.constraint_nn, 
            basic.expert_data,
            p=True\
        ).mean()
        print("Expert => P_aprx(C<=beta) = %.2f, beta = %.2f" % (expert_satisfaction2, basic.beta))
        expert_cdf = compute_cdf(basic.constraint_nn, basic.expert_data)
        agent_cdf = compute_cdf(basic.constraint_nn, basic.agent_data)
        if expert_cdf != None:
            print("Expert => P_empr(C<=beta) = %.2f, beta = %.2f" % (expert_cdf, basic.beta))
        else:
            print("Expert => P_empr(C<=beta) = None, beta = %.2f" % (basic.beta))
        if agent_cdf != None:
            print("Agent => P_empr(C<=beta) = %.2f, beta = %.2f" % (agent_cdf, basic.beta))
        else:
            print("Agent => P_empr(C<=beta) = None, beta = %.2f" % (basic.beta))
    cmse = constraint_mse(true_constraint_function, current_constraint_function)
    if cmse == None:
        print("CMSE: None")
    else:
        print("CMSE: %.2f" % cmse)
    nad = accrual_comparison(basic.expert_data, basic.agent_data)
    if nad == None:
        print("NAD: None")
    else:
        print("NAD: %.2f" % nad)

def create_env():
    if basic.env_name == 'gridworld':
        basic.gridworld_dim = 7
        basic.n_actions = 8
        basic.constraint_fn_input_dim = 2
        basic.beta = 0.99
        basic.discount_factor = 1.0
        basic.ppo_iters = 500
        basic.policy_add_to_mix_every = 250
        r = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
        r[6, 0] = 1.0
        t = [(6, 0)]
        u = [(ui, uj) for ui in [3] for uj in [0, 1, 2, 3]]
        s = [(ui, uj) for ui in [0, 1, 2] for uj in [0, 1]]
        env = tools.environments.GridworldEnvironment(
            start_states=s,
            t=t,
            r=r,
            unsafe_states=u,
            stay_action=False,  # no action to "stay in current cell"
        )
        env = tools.environments.TimeLimit(env, 50)
    elif basic.env_name == 'cartpole':
        basic.constraint_fn_input_dim = 2
        basic.beta = 20
        basic.discount_factor = 0.99
        basic.ppo_iters = 300
        basic.policy_add_to_mix_every = 150
        env = tools.environments.GymEnvironment(
            "CustomCartPole", 
            start_pos=[[-2, 2]], # [[-2.4, -1.15], [1.15, 2.4]]
        )
        env = tools.environments.TimeLimit(env, 200)
    else:
        print('Bad env_name')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)
    env = tools.environments.FollowGymAPI(env)
    return env

def gridworld_imshow(m, fig, ax):
    m = np.array(m).squeeze()
    assert len(m.shape) == 2
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(m, cmap="gray")
    # im.set_clim(0, 1)
    ax.set_xticks(np.arange(m.shape[0]))
    ax.set_yticks(np.arange(m.shape[1]))
    cbar = fig.colorbar(im, cax=cax)

def visualize_constraint(constraint_fn, savefig=None, fig=None, ax=None):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    if basic.env_name == 'gridworld':
        grid_for_action = []
        for a in np.arange(basic.n_actions):
            grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
            for x in np.arange(basic.gridworld_dim):
                for y in np.arange(basic.gridworld_dim):
                    grid[x, y] = constraint_fn(([x, y], a))
            grid_for_action += [grid]
        average_grid = np.mean(grid_for_action, axis=0)
        gridworld_imshow(average_grid, fig, ax)
    elif basic.env_name == 'cartpole':
        a0, a1 = [], []
        for x in np.arange(-2.4, 2.4+0.1, 0.1):
            a0 += [constraint_fn(([x], 0))]
            a1 += [constraint_fn(([x], 1))]
        ax.plot(np.arange(-2.4, 2.4+0.1, 0.1), a0, label="a=0", color='blue')
        ax.plot(np.arange(-2.4, 2.4+0.1, 0.1), a1, label="a=1", color='red')
        ax.legend(loc='best')
        ax.set_ylim(0-0.05, 1+0.05)
        ax.margins(0.05, 0.05)
    else:
        print('Bad env_name')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)
    if savefig != None:
        fig.tight_layout()
        fig.savefig(savefig)

def visualize_accrual(data, savefig=None, fig=None, ax=None):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    if basic.env_name == 'gridworld':
        accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in data:
            for s, a in zip(S, A):
                x, y = s
                accrual[a][x][y] += 1
        accrual = np.mean(accrual, axis=0)
        accrual /= (np.max(accrual)+1e-6)
        gridworld_imshow(accrual, fig, ax)
    elif basic.env_name == 'cartpole':
        accrual = np.zeros_like(np.arange(-2.4, 2.4+0.1, 0.1))
        for S, A in data:
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 49)
                accrual[bin_nbr] += 1
        accrual /= (accrual.max()+1e-6)
        ax.plot(np.arange(-2.4, 2.4+0.1, 0.1), accrual, color='green')
        ax.margins(0.05, 0.05)
    else:
        print('Bad env_name')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)
    if savefig != None:
        fig.tight_layout()
        fig.savefig(savefig)

def current_constraint_function(sa):
    s, a = sa
    if basic.env_name == 'gridworld':
        return basic.constraint_nn(torch.tensor(s, device=basic.device, dtype=torch.float)).detach().cpu()  # Change this depending on the constraint_nn input
    elif basic.env_name == 'cartpole':
        return basic.constraint_nn(torch.tensor([s[0], a], device=basic.device, dtype=torch.float)).detach().cpu()
    else:
        print('Bad env_name')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

def make_nn():
    value_nn = torch.nn.Sequential(
        torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, 1),
    ).to(basic.device)
    policy_nn = torch.nn.Sequential(
        torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.act_n),
    ).to(basic.device)
    return value_nn, policy_nn

def play_episode(env, policy_nn, constraint_fn):
    S, A, R, C = [], [], [], []
    S += [env.reset()]
    done = False
    while not done:
        if not basic.continuous_actions:
            probs = torch.nn.Softmax(dim=-1)(policy_nn(torch.tensor(S[-1], device=basic.device, dtype=torch.float))).view(-1)
            action = np.random.choice(basic.act_n, p=probs.cpu().detach().numpy())
        else:
            print('TODO')
            exit(0)
        A += [action]
        next_state, reward, done, info = env.step(action)
        C += [constraint_fn((S[-1], action))]
        S += [next_state]
        R += [reward]
    return S, A, R, C

def discount(x, invert=False):
    n = len(x)
    g = 0
    d = []
    for i in range(n):
        if not invert:
            g = x[n - 1 - i] + basic.discount_factor * g
            d = [g] + d
        else:
            g = g + x[i] * (basic.discount_factor**i)
            d = d + [g]
    return d

class ReplayBuffer:
    def __init__(self, N):
        self.N = N
        if not basic.continuous_actions:
            self.S = torch.zeros((self.N, basic.obs_n), dtype=torch.float, device=basic.device)
            self.A = torch.zeros((self.N), dtype=torch.long, device=basic.device)
            self.G = torch.zeros((self.N), dtype=torch.float, device=basic.device)
            self.log_probs = torch.zeros((self.N), dtype=torch.float, device=basic.device)
        else:
            print('TODO')
            exit(0)
        self.i = 0
        self.filled = 0

    def add(self, S, A, G, log_probs):
        if not basic.continuous_actions:
            M = S.shape[0]
            self.filled = min(self.filled + M, self.N)
            assert M <= self.N
            for j in range(M):
                self.S[self.i] = S[j, :]
                self.A[self.i] = A[j]
                self.G[self.i] = G[j]
                self.log_probs[self.i] = log_probs[j]
                self.i = (self.i + 1) % self.N
        else:
            print('TODO')
            exit(0)

    def sample(self, n):
        minibatch = random.sample(range(self.filled), min(n, self.filled))
        if not basic.continuous_actions:
            S, A, G, log_probs = [], [], [], []
            for mbi in minibatch:
                s, a, g, lp = self.S[mbi], self.A[mbi], self.G[mbi], self.log_probs[mbi]
                S += [s]
                A += [a]
                G += [g]
                log_probs += [lp]
            return torch.stack(S), torch.stack(A), torch.stack(G), torch.stack(log_probs)
        else:
            print('TODO')
            exit(0)

def sigmoid(x, c):
    return 1./(1.+torch.exp(-c*x))

def sample_gumbel(n, k):
    unif = torch.distributions.Uniform(0,1).sample((n, k))
    g = -torch.log(-torch.log(unif)).to(basic.device)
    return g

def sample_gumbel_softmax(pi, n, temperature):
    k = len(pi)
    g = sample_gumbel(n, k)
    h = (g + torch.log(pi))/temperature
    h_max = h.max(dim=1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y

# p=True means probabilistic constraint
def ppo_penalty(n_epochs, env, policy_nn, value_nn, constraint_fn, additional_fn_condition=None, additional_fn_epoch_end=None, p=False):
    # Additional function to be run when additional function condition is met
    # Additional function takes in current policy_nn and current iteration number
    buffer = ReplayBuffer(basic.replay_buffer_size)
    value_opt = torch.optim.Adam(value_nn.parameters(), lr=basic.learning_rate)
    policy_opt = torch.optim.Adam(policy_nn.parameters(), lr=basic.learning_rate)
    for epoch in range(n_epochs):
        S_e, A_e, G_e, Gc_e, Indices = [], [], [], [], []
        G0_e, Gc0_e = [], []
        max_cost_reached = 0.0
        max_cost_reached_n = 0
        S_e_buf, A_e_buf, G_e_buf = [], [], []
        for episode in range(basic.episodes_per_epoch):
            S, A, R, C = play_episode(env, policy_nn, constraint_fn)
            start_index = len(A_e)
            S_e += S[:-1]  # ignore last state
            A_e += A
            G_e += discount(R)
            G0_e += [float(discount(R)[0])]
            Gc_e += discount(C)
            Gc0_e += [float(discount(C)[0])]
            end_index = len(A_e)
            Indices += [(start_index, end_index)]
            # Only add those experiences to replay buffer which are before the constraint violation
            inverted_discount = torch.tensor(discount(C, invert=True), dtype=torch.float, device=basic.device)
            good_until = len(inverted_discount)
            for idx, item in enumerate(inverted_discount):
                if item >= basic.beta:
                    good_until = idx + 1
                    break
            S_e_buf += S[:-1][:good_until]
            A_e_buf += A[:good_until]
            G_e_buf += discount(R[:good_until])
            if Gc0_e[-1] >= basic.beta:
                max_cost_reached += 1
            max_cost_reached_n += 1
        print("Epoch %d:\tG_avg = %.2f\tGc_avg = %.2f\tMaxCostReached = %.2f"
            % (epoch, np.mean(G0_e), np.mean(Gc0_e), float(max_cost_reached / max_cost_reached_n)))
        S_e = torch.tensor(S_e, dtype=torch.float, device=basic.device)
        A_e = torch.tensor(A_e, dtype=torch.long, device=basic.device)
        G_e = torch.tensor(G_e, dtype=torch.float, device=basic.device)
        Gc_e = torch.tensor(Gc_e, dtype=torch.float, device=basic.device)
        S_e_buf = torch.tensor(S_e_buf, dtype=torch.float, device=basic.device)
        A_e_buf = torch.tensor(A_e_buf, dtype=torch.long, device=basic.device)
        G_e_buf = torch.tensor(G_e_buf, dtype=torch.float, device=basic.device)
        log_probs_e_buf = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e_buf)).gather(1, A_e_buf.view(-1, 1)).view(-1)
        buffer.add(S_e_buf, A_e_buf, G_e_buf, log_probs_e_buf.detach())
        feasibility_opt = torch.optim.Adam(policy_nn.parameters(), lr=basic.learning_rate_feasibility)
        # Penalty update on complete trajectories
        if p:
            Gc0s = []
            loss1_terms = []
            loss2_terms = []
            for start_index, end_index in Indices:
                Gc = Gc_e[start_index:end_index].view(-1)
                if basic.ppo_sigmoid:
                    Gc0s += [sigmoid(Gc[0] - basic.beta, -basic.sigmoid_lambda)]
                else:
                    t = Gc[0] - basic.beta
                    Gc0s += [0.5 - 0.5 * t / torch.sqrt(t*t + basic.small_eps)]
                if not basic.continuous_actions:
                    A_logits = torch.nn.Softmax(dim=-1)(policy_nn(S_e[start_index:end_index])) # batch x act_dim
                    A_diff = torch.stack([sample_gumbel_softmax(pi = A_logits[t, :], n = 1, temperature = 0.01) for t in range(0, end_index-start_index)]).squeeze(1)
                else:
                    print('TODO')
                    exit(0)
                if basic.env_name == 'gridworld':
                    SA_diff = S_e[start_index:end_index] # Change this depending on constraint_nn's input
                elif basic.env_name == 'cartpole':
                    SA_diff = torch.cat([S_e[start_index:end_index, :1], (A_diff @ torch.arange(basic.act_n).float().to(basic.device)).view(-1, 1)], dim=-1)
                else:
                    print("Bad env_name")
                    print('Allowed: %s' % basic.env_names_allowed)
                    exit(0)
                Costs = basic.constraint_nn(SA_diff).view(-1)
                Gc0_diff = discount(Costs)[0]
                if not basic.continuous_actions:
                    log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e[start_index:end_index])).gather(1, A_diff.detach().argmax(dim=-1).long().view(-1, 1)).view(-1)
                    loss1_terms += [Gc0s[-1].detach() * log_probs.sum()]
                else:
                    print('TODO')
                    exit(0)
                if basic.ppo_sigmoid:
                    loss2_terms += [Gc0s[-1].detach() * (1-Gc0s[-1]).detach() * Gc0_diff]
                else:
                    t = Gc[0] - basic.beta
                    factor = basic.small_eps / ((torch.sqrt(t*t + basic.small_eps))**3)
                    loss2_terms += [factor.detach() * Gc0_diff]
            prob_c_tau_leq_beta = torch.mean(torch.stack(Gc0s)).detach()
            loss1 = torch.mean(torch.stack(loss1_terms))
            loss2 = torch.mean(torch.stack(loss2_terms))
            feasibility_opt.zero_grad()
            if basic.ppo_sigmoid:
                feasibility_loss = -1. * (basic.delta > prob_c_tau_leq_beta) * (loss1 - basic.sigmoid_lambda * loss2)
            else:
                feasibility_loss = -1. * (basic.delta > prob_c_tau_leq_beta) * (loss1 - 0.5 * loss2)
            feasibility_loss.backward()
            feasibility_opt.step()
        else:
            for start_index, end_index in Indices:
                Gc = Gc_e[start_index:end_index].view(-1)
                if not basic.continuous_actions:
                    log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e[start_index:end_index])).gather(1, A_e[start_index:end_index].view(-1, 1)).view(-1)
                    feasibility_opt.zero_grad()
                    feasibility_loss = (Gc[0] >= basic.beta) * ((Gc * log_probs).sum())
                    feasibility_loss.backward()
                    feasibility_opt.step()
                else:
                    print('TODO')
                    exit(0)
        # Policy and value update from replay buffer
        for subepoch in range(basic.ppo_subepochs):
            S, A, G, old_log_probs = buffer.sample(basic.minibatch_size)
            if not basic.continuous_actions:
                log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S)).gather(1, A.view(-1, 1)).view(-1)
            else:
                print('TODO')
                exit(0)
            value_opt.zero_grad()
            value_loss = (G - value_nn(S)).pow(2).mean()
            value_loss.backward()
            value_opt.step()
            policy_opt.zero_grad()
            advantages = G - value_nn(S)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - basic.clip_param, 1 + basic.clip_param)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            if not basic.continuous_actions:
                probs_all = torch.nn.Softmax(dim=-1)(policy_nn(S))
                log_probs_all = torch.nn.LogSoftmax(dim=-1)(policy_nn(S))
                entropy = -(probs_all * log_probs_all).sum(1).mean()
            else:
                print('TODO')
                exit(0)
            policy_loss -= basic.entropy_coef * entropy
            policy_loss.backward()
            policy_opt.step()
        # Run additional function at epoch end if condition is met
        # Just in case we need it for later! (and we will)
        if additional_fn_condition != None and additional_fn_epoch_end != None:
            if additional_fn_condition(epoch, policy_nn) == True:
                additional_fn_epoch_end(epoch, policy_nn)

def collect_trajectories(n, env, policy_nn, constraint_fn, only_success=False):
    data = []
    for traj in tqdm.tqdm(range(n)):
        S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        while (not (discount(C)[0] <= basic.beta)) and only_success:
            S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        data += [[S[:-1], A]]
    return data

def convert_to_flow_data(data):
    flow_data = []
    for S, A in data:
        for s, a in zip(S, A):
            if basic.env_name == 'gridworld':
                flow_data += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                flow_data += [[s[0], a]]
            else:
                print("Bad env_name")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
    flow_data = torch.tensor(flow_data, dtype=torch.float, device=basic.device)
    return flow_data

def dissimilarity_wrt_expert(data, mean=True):
    expert_nll_mean, expert_nll_std = basic.expert_nll
    sims = []
    for S, A in data:
        traj_data = []
        for s, a in zip(S, A):
            if basic.env_name == 'gridworld':
                traj_data += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                traj_data += [[s[0], a]]
            else:
                print("Bad env_name")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
        traj_data = torch.tensor(traj_data, dtype=torch.float, device=basic.device)
        traj_nll = -basic.flow.log_probs(traj_data).detach().cpu()
        sims += [(traj_nll > expert_nll_mean + expert_nll_std).float().mean().cpu()]
    if mean:
        return np.mean(sims)
    return torch.tensor(sims, dtype=torch.float, device=basic.device)

def collect_trajectories_mixture(n, env, policy_mixture, weights_mixture, constraint_fn):
    data = []
    value_nn, policy_nn = make_nn()
    normalized_weights_mixture = np.copy(weights_mixture) / np.sum(weights_mixture)
    m = len(weights_mixture)
    for traj in tqdm.tqdm(range(n)):
        chosen_policy_idx = np.random.choice(m, p=normalized_weights_mixture)
        policy_nn.load_state_dict(policy_mixture[chosen_policy_idx])
        S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        data += [[S[:-1], A]]
    return data

def compute_cdf(constraint_nn, data):
    if data == None:
        return None
    Gc0 = []
    for S, A in data:
        input_to_constraint_nn = []
        for s, a in zip(S, A):
            if basic.env_name == 'gridworld':
                input_to_constraint_nn += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                input_to_constraint_nn += [[s[0], a]]
            else:
                print("Bad env_name")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
        input_to_constraint_nn = torch.tensor(input_to_constraint_nn, dtype=torch.float, device=basic.device)
        constraint_values = constraint_nn(input_to_constraint_nn).view(-1)
        Gc0 += [float(discount(constraint_values)[0] <= basic.beta)]
    return np.mean(Gc0)

def compute_current_constraint_value_trajectory(constraint_nn, data, p=False):
    Gc0 = []
    for S, A in data:
        input_to_constraint_nn = []
        for s, a in zip(S, A):
            if basic.env_name == 'gridworld':
                input_to_constraint_nn += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                input_to_constraint_nn += [[s[0], a]]
            else:
                print("Bad env_name")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
        input_to_constraint_nn = torch.tensor(input_to_constraint_nn, dtype=torch.float, device=basic.device)
        constraint_values = constraint_nn(input_to_constraint_nn).view(-1)
        if p:
            # Gc0 += [sigmoid(discount(constraint_values)[0] - basic.beta, -basic.sigmoid_lambda)]
            t = discount(constraint_values)[0] - basic.beta
            Gc0 += [0.5 - 0.5 * t / torch.sqrt(t*t + basic.small_eps)]
        else:
            Gc0 += [discount(constraint_values)[0]]
    return torch.stack(Gc0)

def constraint_function_adjustment(n, constraint_nn, constraint_opt, expert_data, agent_data, p=False):
    losses = []
    for _ in range(n):
        constraint_opt.zero_grad()
        per_traj_dissimilarity = dissimilarity_wrt_expert(agent_data, mean=False)
        per_traj_dissimilarity = per_traj_dissimilarity / per_traj_dissimilarity.sum()
        agent_data_constraint_returns = compute_current_constraint_value_trajectory(constraint_nn, agent_data, p=p)
        expert_data_constraint_returns = compute_current_constraint_value_trajectory(constraint_nn, expert_data, p=p)
        if p:
            # print(agent_data_constraint_returns.shape, per_traj_dissimilarity.shape)
            # print(per_traj_dissimilarity, torch.sum(agent_data_constraint_returns))
            loss1 = (agent_data_constraint_returns * per_traj_dissimilarity).sum() # no minus sign
        else:
            loss1 = -(agent_data_constraint_returns * per_traj_dissimilarity).sum()
        if p:
            # print(expert_data_constraint_returns.shape)
            # print(torch.sum(expert_data_constraint_returns))
            loss2 = -(basic.delta > torch.mean(expert_data_constraint_returns).detach()).float() * (torch.mean(expert_data_constraint_returns))
        else:
            loss2 = ((expert_data_constraint_returns >= basic.beta).float() * (expert_data_constraint_returns - basic.beta)).mean()
        loss = loss1 + basic.alpha * loss2
        loss.backward()
        constraint_opt.step()
        losses += [loss.item()]
    return np.mean(losses)

def condition(epoch, policy_nn):
    if (epoch + 1) % basic.policy_add_to_mix_every == 0:
        return True
    return False

def command(epoch, policy_nn):
    agent_data = collect_trajectories(len(basic.expert_data), basic.env, policy_nn, current_constraint_function)
    basic.policy_mixture += [copy.deepcopy(policy_nn.state_dict())]
    basic.weights_mixture += [dissimilarity_wrt_expert(agent_data)]
    print("Added policy with dissimilarity = %.2f" % basic.weights_mixture[-1])

def generate_expert_data(env, only_success=False, p=False):  # use only_success if the policy isn't great and you just want to get optimal trajectories
    if os.path.exists(basic.expert_data_file):
        return torch.load(basic.expert_data_file)
    value_nn, policy_nn = make_nn()
    if p:
        ppo_penalty(basic.ppo_iters, env, policy_nn, value_nn, basic.true_constraint_function, p=p)        
    else:
        ppo_penalty(basic.ppo_iters, env, policy_nn, value_nn, basic.true_constraint_function)
    expert_data = collect_trajectories(basic.n_trajs, basic.env, policy_nn, basic.true_constraint_function, only_success=only_success)
    torch.save(expert_data, basic.expert_data_file)
    return expert_data

def wasserstein_distance2d(u, v, p='cityblock'):
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape and len(u.shape) == 2)
    dim1, dim2 = u.shape
    assert(p in ['euclidean', 'cityblock'])
    coords = np.zeros((dim1*dim2, 2)).astype('float')
    for i in range(dim1):
        for j in range(dim2):
            coords[i*dim2+j, :] = [i, j]
    d = cdist(coords, coords, p)
    u /= u.sum()
    v /= v.sum()
    return ot.emd2(u.flatten(), v.flatten(), d)

def mse(u, v):
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape)
    return np.mean(np.power(u-v, 2))

def mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        # print(torch.cuda.list_gpu_processes())
        return "%.2f GB free" % (int(torch.cuda.mem_get_info()[0])/(1024*1024*1024))
    else:
        return "Using CPU"
