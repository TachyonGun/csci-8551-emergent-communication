# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:44:52 2021

@author: sierr
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def visualize_world(locations, physical, args, n_agents=None, batch=0, tol=4, t=None, clear_previous=True, save=True, trajectories=[]):

  batch = batch
  player_icon = "o"
  landmark_icons = ["^", "s", "P", "X"]
  colors = ["r", "b", "g", "c", "m", "k"]

  n_entities, _ = locations[batch].shape

  world_dim = args["world_dim"] + tol

  plt.xlim([-world_dim, world_dim])
  plt.ylim([-world_dim, world_dim])

  if n_agents is None:
    list_of_agents = [i for i in range(args["min_agents"])]
  else:
    list_of_agents = [i for i in range(n_agents)]

  for i in reversed(range(n_entities)):
    color_id = int(torch.clone(physical[batch][i, 0]).detach().numpy())
    icon_id = int(torch.clone(physical[batch][i, 1]).detach().numpy())
    icon = player_icon if (i in list_of_agents) else landmark_icons[icon_id]
    color = colors[color_id]
    x, y = locations[batch][i,:].detach().numpy()
    if i in list_of_agents:
      plt.scatter(x, y, c=color, marker=icon, s=world_dim*20, alpha=0.4)
      plt.scatter(x, y, c=color, marker=icon, s=world_dim*5)
    else:
      plt.scatter(x, y, c=color, marker=icon, s=world_dim*15)
    
  if t is not None:
    plt.title(f"t = {t:03d}")

  if len(trajectories) >= 2:
    for i in range(1,len(trajectories)):
      coord_init, coord_last = trajectories[i-1][batch], trajectories[i][batch]
      for j in list_of_agents:
        x_in, y_in = coord_init[j, :]
        x_out, y_out = coord_last[j, :]
        plt.plot([x_in, x_out], [y_in, y_out], linestyle='dotted', c='gray')


  #plt.axis("off")
  plt.xticks([])
  plt.yticks([])
  plt.gca().set_aspect('equal', adjustable='box')
  plt.gcf().set_size_inches(6, 6)
  if clear_previous:
    clear_output()
  if save:
    plt.savefig(f"{t:03d}.png")
  plt.show()

def visualize_episode(timesteps, batch=0, delay=1, clear_previous=True, track_trajectories=True):

  trajectories = []

  for t in range(args["n_timesteps"] + 1):

    if track_trajectories:
      trajectories.append(timesteps[t][0])
    
    print(len(trajectories))
    visualize_world(*timesteps[t], args, t=t, batch=batch, clear_previous=clear_previous, trajectories=trajectories)
    
    time.sleep(delay)

def make_gif(batch=0):
  fp_in = "*.png"
  fp_out = f"movie_{batch}.gif"

  # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
  img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
  img.save(fp=fp_out, format='GIF', append_images=imgs,
          save_all=True, duration=200, loop=0)