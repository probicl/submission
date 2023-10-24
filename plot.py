import os, glob, torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import tools
import basic
import matplotlib
import copy
from matplotlib.patches import Rectangle
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cmr",
    "font.size": 14,
})

if not os.path.exists("output"):
    os.mkdir("output")
if not os.path.exists("output/figures"):
    os.mkdir("output/figures")

# viz = 'gridworld'
viz = 'cartpole'

# figtype = 'setup'
figtype = 'constraints'
# figtype = 'accruals'
# figtype = 'stats'
# figtype = 'histograms'

basic.env_name = viz
basic.true_constraint_function = utils.true_constraint_function

if viz == 'gridworld':

    basic.gridworld_dim = 7
    basic.n_actions = 8
    
    if figtype == 'setup':

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        utils.visualize_constraint(basic.true_constraint_function, fig=fig, ax=ax)
        starts = np.zeros((7, 7))*np.nan
        for xx in range(3):
            for yy in range(2):
                starts[xx][yy] = 0
        ends = np.zeros((7, 7))*np.nan
        ends[6][0] = 0
        cmap1 = plt.cm.Wistia
        cmap1.set_bad((0, 0, 0, 0))
        ax.imshow(starts, cmap=cmap1, alpha=1)
        cmap2 = plt.cm.summer
        cmap2.set_bad((0, 0, 0, 0))
        ax.imshow(ends, cmap=cmap2, alpha=1)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        # ax.set_title('True constraint')
        fig.tight_layout()
        plt.savefig("output/figures/figure1.png")
        plt.show()

    if figtype == 'constraints': 
    
        fig, ax = plt.subplots(1, 4, figsize=(13, 3))
        files = glob.glob("output/gridworld/gridworld_icl/*-cost.pt")
        all_grids = []
        utils.visualize_constraint(basic.true_constraint_function, fig=fig, ax=ax[0])
        true_grid_for_action = []
        for a in np.arange(basic.n_actions):
            true_grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
            for x in np.arange(basic.gridworld_dim):
                for y in np.arange(basic.gridworld_dim):
                    true_grid[x, y] = basic.true_constraint_function(([x, y], a)).item()
            true_grid_for_action += [true_grid]
        true_average_grid = true_grid_for_action # np.mean(true_grid_for_action, axis=0)
        true_grid = true_average_grid
        ax[0].set_xlabel('X-coordinate')
        ax[0].set_ylabel('Y-coordinate')
        ax[0].set_title('True constraint')
        idxnet = 0
        for file in files:
            configuration = tools.data.Configuration.from_json("../ICL-code/ICL/configs/gridworldA.json", {"env":""})
            cost = tools.functions.CostFunction(configuration, i=configuration["i"], h=64, o=1)
            cost.load(file, map_location="cpu")
            cost.Cost.to(basic.device)

            state_action_space = tools.environments.get_state_action_space(
                configuration["env_type"], configuration["env_id"])
            
            # COMMENT/UNCOMMENT for learning policy
            # basic.env = utils.create_env()
            # basic.obs_n = basic.env.observation_space.shape[0]
            # basic.act_n = basic.env.action_space.n
            # basic.env.seed(basic.seed)
            # value_nn, policy_nn = utils.make_nn()
            # basic.constraint_nn = cost.Cost
            # utils.ppo_penalty(basic.ppo_iters, basic.env, policy_nn, value_nn, utils.current_constraint_function)
            # torch.save([value_nn, policy_nn], "output/gridworld_icl/final_policy%d.pt" % idxnet)
            # print("idxnet %d done" % idxnet)
            # idxnet += 1

            # costvalues, costmap = cost.outputs(state_action_space, no_plot=True)
            grid_for_action = []
            for a in np.arange(8):
                grid = np.zeros((7, 7))
                for x in np.arange(7):
                    for y in np.arange(7):
                        grid[x, y] = cost.Cost(torch.tensor([x, y], dtype=torch.float, device=basic.device)).detach().cpu()
                grid_for_action += [grid]
            average_grid = np.mean(grid_for_action, axis=0)
            # costvalues = np.array(costvalues).squeeze()
            # print(np.round(costvalues, 2))
            # exit(0)
            # costvalues = np.mean(costvalues, axis=-1)
            # all_grids += [costvalues]
            all_grids += [average_grid]
        utils.gridworld_imshow(np.mean(all_grids, axis=0), fig, ax[1])
        icl_grid = copy.deepcopy(all_grids) # np.mean(all_grids, axis=0)
        icl_mses = [utils.mse(item1, item2) for item1, item2 in zip(true_grid, icl_grid)]
        # print(icl_mses)
        print("CMSE for ICL: %.2f ± %.2f" % (np.mean(icl_mses), np.std(icl_mses)))
        ax[1].set_xlabel('X-coordinate')
        ax[1].set_ylabel('Y-coordinate')
        ax[1].set_title("Expected constraint (ICL)")
        for di, delta in enumerate([0.5, 0.9]):
            all_grids = []
            for seed in [0,1,2,3,4]:
                f = open("output/gridworld/gridworld_%g/%d/0_log.txt" % (delta, seed))
                constraint_nn, _, _, _ = torch.load("output/gridworld/gridworld_%g/%d/iter_9.pt" % (delta, seed))
                grid_for_action = []
                for a in np.arange(8):
                    grid = np.zeros((7, 7))
                    for x in np.arange(7):
                        for y in np.arange(7):
                            grid[x, y] = constraint_nn(torch.tensor([x, y], dtype=torch.float)).detach().cpu()
                    grid_for_action += [grid]
                average_grid = np.mean(grid_for_action, axis=0)
                all_grids += [average_grid]
                # lines = [item.strip() for item in f.readlines()]
                # for line in lines:
                #     print(line)
            curr_grid = copy.deepcopy(all_grids) # np.mean(all_grids, axis=0)
            utils.gridworld_imshow(np.mean(all_grids, axis=0), fig, ax[di+2])
            ax[di+2].set_xlabel('X-coordinate')
            ax[di+2].set_ylabel('Y-coordinate')
            ax[di+2].set_title("Prob ICL ($\delta$=%g)" % delta)
            curr_mses = [utils.mse(item1, item2) for item1, item2 in zip(true_grid, curr_grid)]
            print("CMSE for Prob ICL delta=%g: %.2f ± %.2f" % (delta, np.mean(curr_mses), np.std(curr_mses)))
        fig.tight_layout()
        plt.savefig("output/figures/figure2.png")
        plt.show()

    elif figtype == 'accruals':

        fig, ax = plt.subplots(1, 4, figsize=(13, 3))
        all_expert_data = []
        for delta in [0.5,0.9]:
            for seed in [0,1,2,3,4]:
                basic.expert_data_file = "output/gridworld/gridworld_%g/%d/expert_data.pt" % (delta, seed)
                basic.expert_data = torch.load(basic.expert_data_file)
                accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
                for S, A in basic.expert_data:
                    for s, a in zip(S, A):
                        x, y = s
                        accrual[a][x][y] += 1
                accrual = np.mean(accrual, axis=0)
                accrual /= (np.max(accrual)+1e-6)
                all_expert_data += [accrual]
        all_expert_data = np.mean(all_expert_data, axis=0)
        utils.gridworld_imshow(all_expert_data, fig, ax[0])
        ax[0].set_xlabel('X-coordinate')
        ax[0].set_ylabel('Y-coordinate')
        ax[0].set_title('Expert accrual')
        files = glob.glob("output/gridworld/gridworld_icl/*-acr.pt")
        all_grids = []
        for file in files:
            all_grids += [torch.load(file)[-1]]
        icl_nads = [utils.wasserstein_distance2d(item1, all_expert_data) for item1 in all_grids]
        print("NAD for ICL: %.2f ± %.2f" % (np.mean(icl_nads), np.std(icl_nads)))
        utils.gridworld_imshow(np.mean(all_grids, axis=0), fig, ax[1])
        ax[1].set_xlabel('X-coordinate')
        ax[1].set_ylabel('Y-coordinate')
        ax[1].set_title("ICL Accrual")
        for di, delta in enumerate([0.5, 0.9]):
            all_grids = []
            for seed in [0,1,2,3,4]:
                # f = open("output/gridworld_%g/%d/0_log.txt" % (delta, seed))
                agent_data, _, _ = torch.load("output/gridworld/gridworld_%g/%d/final_policy.pt" % (delta, seed))
                accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
                for S, A in agent_data:
                    for s, a in zip(S, A):
                        x, y = s
                        accrual[a][x][y] += 1
                accrual = np.mean(accrual, axis=0)
                accrual /= (np.max(accrual)+1e-6)
                all_grids += [accrual]
            # utils.visualize_accrual(np.mean(all_grids, axis=0), fig=fig, ax=ax[di+2])
            curr_nads = [utils.wasserstein_distance2d(item1, all_expert_data) for item1 in all_grids]
            print("NAD for Prob ICL delta=%g: %.2f ± %.2f" % (delta, np.mean(curr_nads), np.std(curr_nads)))
            utils.gridworld_imshow(np.mean(all_grids, axis=0), fig, ax[di+2])
            ax[di+2].set_xlabel('X-coordinate')
            ax[di+2].set_ylabel('Y-coordinate')
            ax[di+2].set_title("Prob ICL Accrual ($\delta$=%g)" % delta)
        fig.tight_layout()
        plt.savefig("output/figures/figure3.png")
        plt.show()

    elif figtype == 'stats':

        for di, delta in enumerate([0.1, 0.5, 0.9]):
            print("\nbeta=0.99, delta=%g" % delta)
            RR, CC, Ea, Ee, Ae, Cmse, Nad = [], [], [], [], [], [], []
            for seed in [0,1,2,3,4]:
                f = open("output/gridworld/gridworld_%g/%d/0_log.txt" % (delta, seed))        
                lines = [line.strip() for line in f.readlines()]
                R = []
                C = []
                Eaprx = []
                Eempr = []
                Aempr = []
                CMSE = []
                NAD = []
                stop = False
                for line in lines:
                    if 'Final results saved' in line:
                        stop = True
                    if "Epoch 499:" in line:
                        R += [line]
                        C += [line]
                    if "Expert => P_aprx" in line:
                        Eaprx += [line]
                    if "Expert => P_empr" in line:
                        Eempr += [line]
                    if "Agent => P_empr" in line:
                        Aempr += [line]
                    if "CMSE:" in line:
                        CMSE += [line]
                    if "NAD:" in line:
                        NAD += [line]
                RR += [float(R[-1].split("=")[1].strip().split()[0])]
                CC += [float(C[-1].split("=")[2].strip().split()[0])]
                Ea += [float(Eaprx[-1].split(",")[0].split("=")[-1].strip())]
                Ee += [float(Eempr[-1].split(",")[0].split("=")[-1].strip())]
                Ae += [float(Aempr[-1].split(",")[0].split("=")[-1].strip())]
                Cmse += [float(CMSE[-1].split(":")[-1].strip())]
                Nad += [float(NAD[-1].split(":")[-1].strip())]
            print("CMSE: %.2f ± %.2f" % (np.mean(Cmse), np.std(Cmse)))
            print("NAD: %.2f ± %.2f" % (np.mean(Nad), np.std(Nad)))
            print("R: %.2f ± %.2f" % (np.mean(RR), np.std(RR)))
            print("C: %.2f ± %.2f" % (np.mean(CC), np.std(CC)))
            print("Expert P_aprx(C <= beta): %.2f ± %.2f" % (np.mean(Ea), np.std(Ea)))
            print("Expert P_empr(C <= beta): %.2f ± %.2f" % (np.mean(Ee), np.std(Ee)))
            print("Agent P_empr(C <= beta): %.2f ± %.2f" % (np.mean(Ae), np.std(Ae)))

    elif figtype == 'histograms':

        import seaborn as sns
        sns.set_style('darkgrid')
        sns.set()

        fig, ax = plt.subplots(1, 3, figsize=(10, 3))

        basic.env = utils.create_env()
        basic.obs_n = basic.env.observation_space.shape[0]
        basic.act_n = basic.env.action_space.n
        files = glob.glob("output/gridworld/gridworld_icl/*-cost.pt")

        CC = []
        Agentcdf = []

        for si, file in enumerate(files):

            configuration = tools.data.Configuration.from_json("../ICL-code/ICL/configs/gridworldA.json", {"env":""})
            cost = tools.functions.CostFunction(configuration, i=configuration["i"], h=64, o=1)
            cost.load(file, map_location="cpu")
            cost.Cost.to(basic.device)
            basic.constraint_nn = cost.Cost

            value_nn, policy_nn = torch.load("output/gridworld/gridworld_icl/final_policy%d.pt" % (si))
            basic.agent_data = utils.collect_trajectories(\
                100, 
                basic.env, 
                policy_nn, 
                utils.current_constraint_function\
            )
            C = utils.compute_current_constraint_value_trajectory(\
                basic.constraint_nn, 
                basic.agent_data,
                p=False\
            )
            agent_cdf = utils.compute_cdf(basic.constraint_nn, basic.agent_data)
            Agentcdf += [agent_cdf]
            CC += list(C.view(-1).detach().cpu().numpy())

        ax[0].hist(CC, bins=20)
        ax[0].axvline(x=np.mean(CC), color='red')
        ax[0].axvline(x=0.99, color='green')
        ax[0].set(ylabel='')
        ax[0].set_title('ICL')
        print("%.2f, %.2f" % (np.mean(Agentcdf), np.std(Agentcdf)))

        for di, delta in enumerate([0.5, 0.9]):
            all_grids = []
            CC = []
            Agentcdf = []
            for seed in [0,1,2,3,4]:
                constraint_nn, _, _, _ = torch.load("output/gridworld/gridworld_%g/%d/iter_9.pt" % (delta, seed))
                constraint_nn.to(basic.device)
                basic.constraint_nn = constraint_nn
                agent_data, value_nn, policy_nn = torch.load("output/gridworld/gridworld_%g/%d/final_policy.pt" % (delta, seed))
                basic.agent_data = utils.collect_trajectories(\
                    100, 
                    basic.env, 
                    policy_nn, 
                    utils.current_constraint_function\
                )
                C = utils.compute_current_constraint_value_trajectory(\
                    basic.constraint_nn, 
                    basic.agent_data,
                    p=False\
                )
                agent_cdf = utils.compute_cdf(basic.constraint_nn, basic.agent_data)
                Agentcdf += [agent_cdf]
                CC += list(C.view(-1).detach().cpu().numpy())
            ax[di+1].hist(CC, bins=20)
            ax[di+1].axvline(x=np.mean(CC), color='red')
            ax[di+1].axvline(x=0.99, color='green')
            ax[di+1].set(ylabel='')
            ax[di+1].set_title('Prob ICL ($\delta$=%g)' % delta)
            print("%.2f, %.2f, %g" % (np.mean(Agentcdf), np.std(Agentcdf), delta))

        fig.tight_layout()
        plt.savefig("output/figures/figure4.png")
        plt.show()        

elif viz == 'cartpole':

    if figtype == 'setup':

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        utils.visualize_constraint(basic.true_constraint_function, fig=fig, ax=ax)
        ax.add_patch(Rectangle((-2, 0.05), 4, 0.2, color='#e4ff7a', alpha=0.5))
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Constraint value')
        # ax.set_title('True constraint')
        fig.tight_layout()
        plt.savefig("output/figures/cartpole_figure1.png")
        plt.show()

    if figtype == 'constraints':

        fig, ax = plt.subplots(1, 4, figsize=(13, 3))
        utils.visualize_constraint(basic.true_constraint_function, fig=fig, ax=ax[0])
        ax[0].set_xlabel('X-coordinate')
        ax[0].set_ylabel('Constraint value')
        ax[0].set_title("True constraint")
        # files = glob.glob("output/cartpole/cartpole_icl/*-cost.pt")
        all_grids_0, all_grids_1 = [], []
        # for file in files:
        for seed in [0,1,2,3,4]:
            # configuration = tools.data.Configuration.from_json("../ICL-code/ICL/configs/cartpoleM.json", {"env":""})
            # cost = tools.functions.CostFunction(configuration, i=configuration["i"], h=64, o=1)
            # cost.load(file, map_location="cpu")
            # state_action_space = tools.environments.get_state_action_space(
                # configuration["env_type"], configuration["env_id"])
            # costvalues, costmap = cost.outputs(state_action_space, no_plot=True)
            # costvalues = np.array(costvalues).squeeze()
            f = open("output/cartpole/icl/%d/0_log.txt" % (seed))
            # all_grids_0 += [costvalues[0, :]]
            # all_grids_1 += [costvalues[1, :]]
            constraint_nn, _, _, _ = torch.load("output/cartpole/icl/%d/iter_9.pt" % (seed))
            constraint_nn = constraint_nn.cpu()
            a0, a1 = [], []
            for x in np.arange(-2.4, 2.4+0.1, 0.1):
                a0 += [constraint_nn(torch.tensor([x, 0], dtype=torch.float)).detach().cpu().item()]
                a1 += [constraint_nn(torch.tensor([x, 1], dtype=torch.float)).detach().cpu().item()]
            vals = np.array([a0, a1])
            all_grids_0 += [a0]
            all_grids_1 += [a1]
        ax[1].plot(np.arange(-2.4, 2.4+0.1, 0.1), np.mean(all_grids_0, axis=0), label="a=0", color='blue')
        ax[1].plot(np.arange(-2.4, 2.4+0.1, 0.1), np.mean(all_grids_1, axis=0), label="a=1", color='red')
        ax[1].legend(loc='best')
        ax[1].set_ylim(0-0.05, 1+0.05)
        ax[1].margins(0.05, 0.05)
        ax[1].set_xlabel('X-coordinate')
        ax[1].set_ylabel('Constraint value')
        ax[1].set_title("Expected constraint (ICL)")
        for di, delta in enumerate([0.5, 0.9]):
            all_grids_0, all_grids_1 = [], []
            for seed in [0,1,2,3,4]:
                f = open("output/cartpole/ipcl_%g/%d/0_log.txt" % (delta, seed))
                try:
                    constraint_nn, _, _, _ = torch.load("output/cartpole/ipcl_%g/%d/iter_9.pt" % (delta, seed))
                except:
                    print("missed seed %d for delta %.2f" % (seed, delta))
                    continue
                constraint_nn = constraint_nn.cpu()
                a0, a1 = [], []
                for x in np.arange(-2.4, 2.4+0.1, 0.1):
                    a0 += [constraint_nn(torch.tensor([x, 0], dtype=torch.float)).detach().cpu().item()]
                    a1 += [constraint_nn(torch.tensor([x, 1], dtype=torch.float)).detach().cpu().item()]
                vals = np.array([a0, a1])
                all_grids_0 += [a0]
                all_grids_1 += [a1]
                # lines = [item.strip() for item in f.readlines()]
                # for line in lines:
                #     print(line)
            ax[di+2].plot(np.arange(-2.4, 2.4+0.1, 0.1), np.mean(all_grids_0, axis=0), label="a=0", color='blue')
            ax[di+2].plot(np.arange(-2.4, 2.4+0.1, 0.1), np.mean(all_grids_1, axis=0), label="a=1", color='red')
            ax[di+2].legend(loc='best')
            ax[di+2].set_ylim(0-0.05, 1+0.05)
            ax[di+2].margins(0.05, 0.05)
            ax[di+2].set_xlabel('X-coordinate')
            ax[di+2].set_ylabel('Constraint value')
            ax[di+2].set_title("Prob ICL Accrual ($\delta$=%g)" % delta)
        fig.tight_layout()
        plt.savefig("output/figures/cartpole_figure2.png")
        plt.show()
    
    elif figtype == 'accruals':

        fig, ax = plt.subplots(1, 4, figsize=(13, 3))
        all_expert_data = []
        basic.expert_data_file = "output/cartpole/icl_0/expert_data.pt"
        basic.expert_data = torch.load(basic.expert_data_file)
        accrual = np.zeros_like(np.arange(-2.4, 2.4+0.1, 0.1))
        for S, A in basic.expert_data:
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 49)
                accrual[bin_nbr] += 1
        accrual /= (accrual.max()+1e-6)
        ax[0].plot(np.arange(-2.4, 2.4+0.1, 0.1), accrual, color='green')
        ax[0].margins(0.05, 0.05)
        ax[0].set_xlabel('X-coordinate')
        ax[0].set_ylabel('Normalized accrual')
        ax[0].set_title('Expert accrual')
        all_grids = []
        for seed in [0,1,2,3,4]:
            agent_data, _, _ = torch.load("output/cartpole/icl/%d/final_policy.pt" % seed)
            accrual = np.zeros_like(np.arange(-2.4, 2.4+0.1, 0.1))
            for S, A in agent_data:
                for s, a in zip(S, A):
                    bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 49)
                    accrual[bin_nbr] += 1
            accrual /= (accrual.max()+1e-6)
            all_grids += [accrual]
        ax[1].plot(np.arange(-2.4, 2.4+0.1, 0.1), np.mean(all_grids, axis=0), color='green')
        ax[1].margins(0.05, 0.05)
        ax[1].set_xlabel('X-coordinate')
        ax[1].set_ylabel('Normalized accrual')
        ax[1].set_title('ICL accrual')
        for di, delta in enumerate([0.5, 0.9]):
            all_grids = []
            for seed in [0,1,2,3,4]:
                # f = open("output/gridworld_%g/%d/0_log.txt" % (delta, seed))
                try:
                    agent_data, _, _ = torch.load("output/cartpole/ipcl_%g/%d/final_policy.pt" % (delta, seed))
                except:
                    print("missed seed %d for delta %.2f" % (seed, delta))
                    continue
                accrual = np.zeros_like(np.arange(-2.4, 2.4+0.1, 0.1))
                for S, A in agent_data:
                    for s, a in zip(S, A):
                        bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 49)
                        accrual[bin_nbr] += 1
                accrual /= (accrual.max()+1e-6)
                all_grids += [accrual]
            ax[di+2].plot(np.arange(-2.4, 2.4+0.1, 0.1), np.mean(all_grids, axis=0), color='green')
            ax[di+2].margins(0.05, 0.05)
            ax[di+2].set_xlabel('X-coordinate')
            ax[di+2].set_ylabel('Normalized accrual')
            ax[di+2].set_title("Prob ICL Accrual ($\delta$=%g)" % delta)
        fig.tight_layout()
        plt.savefig("output/figures/cartpole_figure3.png")
        plt.show()

    elif figtype == 'stats':

        for di, delta in enumerate([-1, 0.5, 0.9]):
            if delta == -1:
                print("\nbeta=20 (icl)")
            else:
                print("\nbeta=20, delta=%g" % delta)
            RR, CC, Ea, Ee, Ae, Cmse, Nad = [], [], [], [], [], [], []
            for seed in [0,1,2,3,4]:
                if delta == -1:
                    f = open("output/cartpole/icl/%d/0_log.txt" % (seed))        
                else:
                    f = open("output/cartpole/ipcl_%g/%d/0_log.txt" % (delta, seed))        
                lines = [line.strip() for line in f.readlines()]
                R = []
                C = []
                Eaprx = []
                Eempr = []
                Aempr = []
                CMSE = []
                NAD = []
                stop = False
                for line in lines:
                    if 'Final results saved' in line:
                        stop = True
                    if "Epoch 299:" in line:
                        R += [line]
                        C += [line]
                    if "Expert => P_aprx" in line:
                        Eaprx += [line]
                    if "Expert => P_empr" in line:
                        Eempr += [line]
                    if "Agent => P_empr" in line:
                        Aempr += [line]
                    if "CMSE:" in line:
                        CMSE += [line]
                    if "NAD:" in line:
                        NAD += [line]
                RR += [float(R[-1].split("=")[1].strip().split()[0])]
                CC += [float(C[-1].split("=")[2].strip().split()[0])]
                Ea += [float(Eaprx[-1].split(",")[0].split("=")[-1].strip())]
                Ee += [float(Eempr[-1].split(",")[0].split("=")[-1].strip())]
                Ae += [float(Aempr[-1].split(",")[0].split("=")[-1].strip())]
                Cmse += [float(CMSE[-1].split(":")[-1].strip())]
                Nad += [float(NAD[-1].split(":")[-1].strip())]
            print("CMSE: %.2f ± %.2f" % (np.mean(Cmse), np.std(Cmse)))
            print("NAD: %.2f ± %.2f" % (np.mean(Nad), np.std(Nad)))
            print("R: %.2f ± %.2f" % (np.mean(RR), np.std(RR)))
            print("C: %.2f ± %.2f" % (np.mean(CC), np.std(CC)))
            print("Expert P_aprx(C <= beta): %.2f ± %.2f" % (np.mean(Ea), np.std(Ea)))
            print("Expert P_empr(C <= beta): %.2f ± %.2f" % (np.mean(Ee), np.std(Ee)))
            print("Agent P_empr(C <= beta): %.2f ± %.2f" % (np.mean(Ae), np.std(Ae)))

    elif figtype == 'histograms':

        import seaborn as sns
        sns.set_style('darkgrid')
        sns.set()

        fig, ax = plt.subplots(1, 3, figsize=(10, 3))

        basic.env = utils.create_env()
        basic.obs_n = basic.env.observation_space.shape[0]
        basic.act_n = basic.env.action_space.n
        # files = glob.glob("output/cartpole/icl/*-cost.pt")

        CC = []
        Agentcdf = []

        # for si, file in enumerate(files):
        for si, seed in enumerate([0,1,2,3,4]):

            # configuration = tools.data.Configuration.from_json("../ICL-code/ICL/configs/gridworldA.json", {"env":""})
            # cost = tools.functions.CostFunction(configuration, i=configuration["i"], h=64, o=1)
            # cost.load(file, map_location="cpu")
            # cost.Cost.to(basic.device)
            # basic.constraint_nn = cost.Cost

            constraint_nn, _, _, _ = torch.load("output/cartpole/icl/%d/iter_9.pt" % (seed))
            basic.constraint_nn = constraint_nn

            _, value_nn, policy_nn = torch.load("output/cartpole/icl/%d/final_policy.pt" % seed)
            basic.agent_data = utils.collect_trajectories(\
                100, 
                basic.env, 
                policy_nn, 
                utils.current_constraint_function\
            )
            C = utils.compute_current_constraint_value_trajectory(\
                basic.constraint_nn, 
                basic.agent_data,
                p=False\
            )
            agent_cdf = utils.compute_cdf(basic.constraint_nn, basic.agent_data)
            Agentcdf += [agent_cdf]
            CC += list(C.view(-1).detach().cpu().numpy())

        ax[0].hist(CC, bins=20)
        ax[0].axvline(x=np.mean(CC), color='red')
        ax[0].axvline(x=20, color='green')
        ax[0].set(ylabel='')
        ax[0].set_title('ICL')
        print("%.2f, %.2f" % (np.mean(Agentcdf), np.std(Agentcdf)))

        for di, delta in enumerate([0.5, 0.9]):
            all_grids = []
            CC = []
            Agentcdf = []
            for seed in [0,1,2,3,4]:
                try:
                    constraint_nn, _, _, _ = torch.load("output/cartpole/ipcl_%g/%d/iter_9.pt" % (delta, seed))
                    constraint_nn.to(basic.device)
                    basic.constraint_nn = constraint_nn
                    _, value_nn, policy_nn = torch.load("output/cartpole/ipcl_%g/%d/final_policy.pt" % (delta, seed))
                    basic.agent_data = utils.collect_trajectories(\
                        100, 
                        basic.env, 
                        policy_nn, 
                        utils.current_constraint_function\
                    )
                    C = utils.compute_current_constraint_value_trajectory(\
                        basic.constraint_nn, 
                        basic.agent_data,
                        p=False\
                    )
                    agent_cdf = utils.compute_cdf(basic.constraint_nn, basic.agent_data)
                    Agentcdf += [agent_cdf]
                    CC += list(C.view(-1).detach().cpu().numpy())
                except:
                    print("missed seed %d for delta %.2f" % (seed, delta))
                    continue
            ax[di+1].hist(CC, bins=20)
            ax[di+1].axvline(x=np.mean(CC), color='red')
            ax[di+1].axvline(x=20, color='green')
            ax[di+1].set(ylabel='')
            ax[di+1].set_title('Prob ICL ($\delta$=%g)' % delta)
            print("%.2f, %.2f, %g" % (np.mean(Agentcdf), np.std(Agentcdf), delta))

        fig.tight_layout()
        plt.savefig("output/figures/cartpole_figure4.png")
        plt.show()        