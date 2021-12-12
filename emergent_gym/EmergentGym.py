import gym

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 22:17:11 2021

@author: sierr
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import os
import glob
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

"""
    The GameModule takes in all actions(movement, utterance, goal prediction)
    of all agents for a given timestep and returns the total cost for that
    timestep.

    Game consists of:
        -num_agents (scalar)
        -num_landmarks (scalar)
        -locations: [num_agents + num_landmarks, 2]
        -physical: [num_agents + num_landmarks, entity_embed_size]
        -utterances: [num_agents, vocab_size]
        -goals: [num_agents, goal_size]
        -location_observations: [num_agents, num_agents + num_landmarks, 2]
        -memories
            -utterance: [num_agents, num_agents, memory_size]
            -physical:[num_agents, num_agents + num_landmarks, memory_size]
            -action: [num_agents, memory_size]

        config needs: -batch_size, -using_utterances, -world_dim, -vocab_size, -memory_size, -num_colors -num_shapes
"""


class EmergentGym(gym.Env):

    def __init__(self, config, num_agents, num_landmarks, args, collect_state_history=True, seed=None):
        super(EmergentGym, self).__init__()

        self.timesteps = []

        if seed is not None:
            torch.manual_seed(seed)
        self.args = args
        self.collect_state_history = collect_state_history
        self.batch_size = config.batch_size  # scalar: num games in this batch
        self.using_utterances = config.use_utterances  # bool: whether current batch allows utterances
        self.using_cuda = config.use_cuda
        self.num_agents = num_agents  # scalar: number of agents in this batch
        self.num_landmarks = num_landmarks  # scalar: number of landmarks in this batch
        self.num_entities = self.num_agents + self.num_landmarks  # type: int

        if self.using_cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        locations = torch.rand(self.batch_size, self.num_entities, 2) * config.world_dim
        colors = (torch.rand(self.batch_size, self.num_entities, 1) * config.num_colors).floor()
        shapes = (torch.rand(self.batch_size, self.num_entities, 1) * config.num_shapes).floor()

        goal_agents = self.Tensor(self.batch_size, self.num_agents, 1)
        goal_entities = (torch.rand(self.batch_size, self.num_agents,
                                    1) * self.num_landmarks).floor().long() + self.num_agents
        goal_locations = self.Tensor(self.batch_size, self.num_agents, 2)

        if self.using_cuda:
            locations = locations.cuda()
            colors = colors.cuda()
            shapes = shapes.cuda()
            goal_entities = goal_entities.cuda()

        # [batch_size, num_entities, 2]
        self.locations = Variable(locations)
        # [batch_size, num_entities, 2]
        self.physical = Variable(torch.cat((colors, shapes), 2).float())

        # TODO: Bad for loop?
        for b in range(self.batch_size):
            goal_agents[b] = torch.randperm(self.num_agents)[:, None]  # expanded with dummy axis

        for b in range(self.batch_size):
            goal_locations[b] = self.locations.data[b][goal_entities[b].squeeze()]

        # [batch_size, num_agents, 3]
        self.goals = Variable(torch.cat((goal_locations, goal_agents), 2))
        goal_agents = Variable(goal_agents)

        if self.using_cuda:
            self.memories = {
                "physical": Variable(
                    torch.zeros(self.batch_size, self.num_agents, self.num_entities, config.memory_size).cuda()),
                "action": Variable(torch.zeros(self.batch_size, self.num_agents, config.memory_size).cuda())}
        else:
            self.memories = {
                "physical": Variable(
                    torch.zeros(self.batch_size, self.num_agents, self.num_entities, config.memory_size)),
                "action": Variable(torch.zeros(self.batch_size, self.num_agents, config.memory_size))}

        if self.using_utterances:
            if self.using_cuda:
                self.utterances = Variable(torch.zeros(self.batch_size, self.num_agents, config.vocab_size).cuda())
                self.memories["utterance"] = Variable(
                    torch.zeros(self.batch_size, self.num_agents, self.num_agents, config.memory_size).cuda())
            else:
                self.utterances = Variable(torch.zeros(self.batch_size, self.num_agents, config.vocab_size))
                self.memories["utterance"] = Variable(
                    torch.zeros(self.batch_size, self.num_agents, self.num_agents, config.memory_size))

        agent_baselines = self.locations[:, :self.num_agents, :]

        sort_idxs = torch.sort(self.goals[:, :, 2])[1]
        self.sorted_goals = Variable(self.Tensor(self.goals.size()))
        # TODO: Bad for loop?
        for b in range(self.batch_size):
            self.sorted_goals[b] = self.goals[b][sort_idxs[b]]
        self.sorted_goals = self.sorted_goals[:, :, :2]

        # [batch_size, num_agents, num_entities, 2]
        self.observations = self.locations.unsqueeze(1) - agent_baselines.unsqueeze(2)

        new_obs = self.goals[:, :, :2] - agent_baselines

        # [batch_size, num_agents, 2] [batch_size, num_agents, 1]
        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)
        if self.collect_state_history:
            self.timesteps.append(self.return_state())

    def return_state(self):
        return [self.locations, self.physical, self.utterances]

    def log_state(self):
        self.timesteps.append(self.return_state())

    """
    Updates game state given all movements and utterances and returns accrued cost
        - movements: [batch_size, num_agents, config.movement_size]
        - utterances: [batch_size, num_agents, config.utterance_size]
        - goal_predictions: [batch_size, num_agents, num_agents, config.goal_size]
    Returns:
        - scalar: total cost of all games in the batch

    """

    def step(self, action):
        movements = action['movements']
        utterances = action['utterances']
        goal_predictions = action['goal_predictions']

        self.locations = self.locations + movements
        agent_baselines = self.locations[:, :self.num_agents]
        self.observations = self.locations.unsqueeze(1) - agent_baselines.unsqueeze(2)
        new_obs = self.goals[:, :, :2] - agent_baselines
        goal_agents = self.goals[:, :, 2].unsqueeze(2)
        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)

        if self.using_utterances:
            self.utterances = utterances

            # Update state history
            if self.collect_state_history:
                self.timesteps.append(self.return_state())
            # self.log_state()
            return self.compute_cost(movements, goal_predictions, utterances)
        else:
            # Update state history
            if self.collect_state_history:
                self.timesteps.append(self.return_state())
            self.log_state()
            return self.compute_cost(movements, goal_predictions)

    def compute_cost(self, movements, goal_predictions, utterances=None):
        physical_cost = self.compute_physical_cost()
        movement_cost = self.compute_movement_cost(movements)
        goal_pred_cost = self.compute_goal_pred_cost(goal_predictions)
        return physical_cost + goal_pred_cost + movement_cost

    """
    Computes the total cost agents get from being near their goals
    agent locations are stored as [batch_size, num_agents + num_landmarks, entity_embed_size]
    """

    def compute_physical_cost(self):
        return 2 * torch.sum(
            torch.sqrt(
                torch.sum(
                    torch.pow(
                        self.locations[:, :self.num_agents, :] - self.sorted_goals,
                        2),
                    -1)
            )
        )

    """
    Computes the total cost agents get from predicting others' goals
    goal_predictions: [batch_size, num_agents, num_agents, goal_size]
    goal_predictions[., a_i, a_j, :] = a_i's prediction of a_j's goal with location relative to a_i
    We want:
        real_goal_locations[., a_i, a_j, :] = a_j's goal with location relative to a_i
    We have:
        goals[., a_j, :] = a_j's goal with absolute location
        observed_goals[., a_j, :] = a_j's goal with location relative to a_j
    Which means we want to build an observed_goals-like tensor but relative to each agent
        real_goal_locations[., a_i, a_j, :] = goals[., a_j, :] - locations[a_i]


    """

    def compute_goal_pred_cost(self, goal_predictions):
        relative_goal_locs = self.goals.unsqueeze(1)[:, :, :, :2] - self.locations.unsqueeze(2)[:, :self.num_agents, :,
                                                                    :]
        goal_agents = self.goals.unsqueeze(1)[:, :, :, 2:].expand_as(relative_goal_locs)[:, :, :, -1:]
        relative_goals = torch.cat((relative_goal_locs, goal_agents), dim=3)
        return torch.sum(
            torch.sqrt(
                torch.sum(
                    torch.pow(
                        goal_predictions - relative_goals,
                        2),
                    -1)
            )
        )

    """
    Computes the total cost agents get from moving
    """

    def compute_movement_cost(self, movements):
        return torch.sum(torch.sqrt(torch.sum(torch.pow(movements, 2), -1)))

    def get_avg_agent_to_goal_distance(self):
        return torch.sum(
            torch.sqrt(
                torch.sum(
                    torch.pow(
                        self.locations[:, :self.num_agents, :] - self.sorted_goals,
                        2),
                    -1)
            )
        )

    def reset(self):
        pass

    def visualize_world_and_vocab(self, locations, physical, args,
                                  n_agents=None,
                                  batch=0,
                                  tol=4,
                                  t=None,
                                  clear_previous=True,
                                  save=True,
                                  trajectories=[],
                                  vocab=None,
                                  utterances=None,
                                  show_speech_bubble=True,
                                  show_first_quadrant=True,
                                  return_plot=False,
                                  filename="",
                                  show_plot=True,
                                  epoch=None):

        batch = batch
        player_icon = "o"
        landmark_icons = ["^", "s", "P", "X"]
        colors = ["r", "b", "g", "c", "m", "k"]

        n_entities = locations[batch].shape[0]
        world_dim = args["world_dim"] + tol

        if utterances is None:
           show_speech_bubble=False
           f, (a0) = plt.subplots(1, 1)
        else:
           f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

        if not show_first_quadrant:
            a0.set_xlim([-world_dim, world_dim])
            a0.set_ylim([-world_dim, world_dim])
        else:
            a0.set_xlim([0, world_dim])
            a0.set_ylim([0, world_dim])

        if epoch is not None:
            f.suptitle(f"epoch: {epoch:03d}")

        if n_agents is None:
            list_of_agents = [i for i in range(args["min_agents"])]
        else:
            list_of_agents = [i for i in range(n_agents)]

        a0.set_aspect(1)

        for i in reversed(range(n_entities)):
            color_id = int(torch.clone(physical[batch][i, 0]).cpu().detach().numpy())
            icon_id = int(torch.clone(physical[batch][i, 1]).cpu().detach().numpy())
            icon = player_icon if (i in list_of_agents) else landmark_icons[icon_id]
            color = colors[color_id]
            x, y = torch.clone(locations[batch][i, :]).cpu().detach().numpy()
            if i in list_of_agents:
                a0.scatter(x, y, c=color, marker=icon, s=world_dim * 20, alpha=0.4)
                a0.scatter(x, y, c=color, marker=icon, s=world_dim * 5)
                if not show_first_quadrant:
                    if show_speech_bubble and x < world_dim and y < world_dim and x > -world_dim and y > -world_dim:
                        utter = np.argmax(torch.clone(utterances[batch][i]).cpu().detach().numpy())
                        a0.text(x - world_dim / 15, y - world_dim / 15, f"[{utter:02d}]")
                else:
                    if show_speech_bubble and x < world_dim and y < world_dim and x > 0 and y > 0:
                        utter = np.argmax(torch.clone(utterances[batch][i]).cpu().detach().numpy())
                        a0.text(x - world_dim / 15, y - world_dim / 15, f"[{utter:02d}]")
            else:
                a0.scatter(x, y, c=color, marker=icon, s=world_dim * 15)

        if t is not None:
            # plt.title(f"t = {t:03d}")
            pass

        if len(trajectories) >= 2:
            for i in range(1, len(trajectories)):
                coord_init, coord_last = trajectories[i - 1][batch].clone().detach(), trajectories[i][
                    batch].clone().detach()
                for j in list_of_agents:
                    x_in, y_in = coord_init[j, :]
                    x_out, y_out = coord_last[j, :]
                    a0.plot([x_in, x_out], [y_in, y_out], linestyle='dotted', c='gray')

        if utterances is not None:
            a1.imshow(torch.clone(utterances[batch]).detach().numpy(), cmap='cividis')
            a1.set_xticks([])
            a1.set_yticks([])

        # plt.axis("off")
        a0.set_xticks([])
        a0.set_yticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gcf().set_size_inches(5, 6)
        plt.subplots_adjust(wspace=-1, hspace=-.25)

        if save:
            plt.savefig(f"{filename}{t:03d}.png", bbox_inches='tight', pad_inches=0.1)
        if clear_previous:
            plt.clf()
        if return_plot:
            return f
        if show_plot:
            plt.show()

    def make_gif(self, batch=0, filename_template="./images/image", filename_save="./images", show_gif=True):
        fp_in = filename_template + "*.png"
        fp_out = f"{filename_save}movie_{batch}.gif"

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)
        # TODO
        # if show_gif:
        #    os.startfile(fp_out)

    def render(self, mode="human"):
        self.visualize_world_and_vocab(self.locations, self.physical, self.args, n_agents=self.num_agents,
                                       batch=0,
                                       tol=4,
                                       t=None,
                                       clear_previous=False,
                                       save=False,
                                       trajectories=[],
                                       vocab=None,
                                       utterances=self.utterances,
                                       show_speech_bubble=True,
                                       show_first_quadrant=True,
                                       return_plot=False,
                                       filename="image.png",
                                       show_plot=True)

    def render_episode(self, epoch=0, show_utterances=False, mode="human"):

        delay = 1
        trajectories = []

        if not os.path.exists(f"./images/epoch-{epoch}/episodes"):
            os.makedirs(f"./images/epoch-{epoch}/episodes")

        for t in range(len(self.timesteps)):
            trajectories.append(self.timesteps[t][0])

            locs, phys = self.timesteps[t][0], self.timesteps[t][1]
            utterances = self.timesteps[t][2] if show_utterances else None
            self.visualize_world_and_vocab(locs, phys, self.args, t=t, trajectories=trajectories, utterances=utterances,
                                           filename=f"./images/epoch-{epoch}/episodes/image", show_plot=False,epoch=epoch)

        self.make_gif(batch=0, filename_template=f"./images/epoch-{epoch}/episodes/",
                      filename_save=f"./images/epoch-{epoch}/")

    def render_episode_grid(self, epoch=0, show_utterances=False, title=None):

            if not os.path.exists(f"./images/epoch-{epoch}/episodes"):
                os.makedirs(f"./images/epoch-{epoch}/episodes")
            if not os.path.exists(f"./images/epoch-{epoch}/comp"):
                os.makedirs(f"./images/epoch-{epoch}/comp")

            for batch in range(8):

                filename = f"{batch}_"
                trajectories = []

                for t in range(len(self.timesteps)):

                    trajectories.append(self.timesteps[t][0])

                    locs, phys = self.timesteps[t][0], self.timesteps[t][1]

                    utterances = self.timesteps[t][2]
                    utterances = self.timesteps[t][2] if show_utterances else None
                    self.visualize_world_and_vocab(locs, phys, self.args, t=t, trajectories=trajectories, batch=batch,
                                                   utterances=utterances,
                                                   filename=f"./images/epoch-{epoch}/episodes/{batch}_", show_plot=False,
                                                   epoch=None)
                    plt.close()

            for t in range(len(self.timesteps)):
                fig, axs = plt.subplots(2, 4)
                row = 0
                col = 0
                for batch in range(8):

                    img = plt.imread(f"./images/epoch-{epoch}/episodes/{batch}_{t:03d}.png")

                    axs[row, col].imshow(img)
                    axs[row, col].axis('off')
                    col += 1
                    if col == 4:
                        row = 1
                        col = 0

                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.gcf().set_size_inches(24, 12)
                plt.tight_layout()

                if title is not None:
                    # plt.title(title)
                    plt.suptitle(title, fontsize=40, y=1.05)

                plt.savefig(f"./images/epoch-{epoch}/comp/comp_{t:03d}.png")
                plt.close()

            self.make_gif(batch=0, filename_template=f"./images/epoch-{epoch}/comp/",
                          filename_save=f"./images/epoch-{epoch}/")
