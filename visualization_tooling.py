import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def visualize_world(locations, 
                    physical, 
                    args, 
                    n_agents=None, 
                    batch=0, 
                    tol=4, 
                    t=None, 
                    clear_previous=True, 
                    save=True, 
                    trajectories=[]):

  batch = batch
  player_icon = "o"
  landmark_icons = ["^", "s", "P", "X"]
  colors = ["r", "b", "g", "c", "m", "k"]

  n_entities = locations[batch].shape[0]

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

def visualize_episode(timesteps, batch=0, delay=1, clear_previous=True, track_trajectories=True, plot_utterances=False):

  trajectories = []

  for t in range(args["n_timesteps"] + 1):

    if track_trajectories:
      trajectories.append(timesteps[t][0])

    locs, phys = timesteps[t][0], timesteps[t][1]
    if plot_utterances:
      utterances = timesteps[t][2]
      visualize_world_and_vocab(locs, phys, args, t=t, batch=batch, clear_previous=clear_previous, trajectories=trajectories, utterances=utterances)
    else:
      visualize_world(locs, phys, args, t=t, batch=batch, clear_previous=clear_previous, trajectories=trajectories)
    time.sleep(delay)

def make_gif(batch=0, filename_template="", filename_save=""):
  fp_in = filename_template+"*.png"
  fp_out = f"{filename_save}movie_{batch}.gif"

  # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
  img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
  img.save(fp=fp_out, format='GIF', append_images=imgs,
          save_all=True, duration=200, loop=0)
  
  
def visualize_world_and_vocab(locations, physical, args, 
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
                              show_plot=True):

  batch = batch
  player_icon = "o"
  landmark_icons = ["^", "s", "P", "X"]
  colors = ["r", "b", "g", "c", "m", "k"]

  n_entities = locations[batch].shape[0]

  world_dim = args["world_dim"] + tol

  f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

  if not show_first_quadrant:
    a0.set_xlim([-world_dim, world_dim])
    a0.set_ylim([-world_dim, world_dim])
  else:
    a0.set_xlim([0, world_dim])
    a0.set_ylim([0, world_dim])

  if n_agents is None:
    list_of_agents = [i for i in range(args["min_agents"])]
  else:
    list_of_agents = [i for i in range(n_agents)]


  a0.set_aspect(1)

  for i in reversed(range(n_entities)):
    color_id = int(torch.clone(physical[batch][i, 0]).detach().numpy())
    icon_id = int(torch.clone(physical[batch][i, 1]).detach().numpy())
    icon = player_icon if (i in list_of_agents) else landmark_icons[icon_id]
    color = colors[color_id]
    x, y = locations[batch][i,:].detach().numpy()
    if i in list_of_agents:
      a0.scatter(x, y, c=color, marker=icon, s=world_dim*20, alpha=0.4)
      a0.scatter(x, y, c=color, marker=icon, s=world_dim*5)
      if not show_first_quadrant:
        if show_speech_bubble and x < world_dim and y < world_dim and x > -world_dim and y > -world_dim:
            utter = np.argmax(torch.clone(utterances[batch][i]).detach().numpy())
            a0.text(x-world_dim/15, y-world_dim/15, f"[{utter:02d}]")
      else: 
        if show_speech_bubble and x < world_dim and y < world_dim and x > 0 and y > 0:
            utter = np.argmax(torch.clone(utterances[batch][i]).detach().numpy())
            a0.text(x-world_dim/15, y-world_dim/15, f"[{utter:02d}]")
    else:
      a0.scatter(x, y, c=color, marker=icon, s=world_dim*15)
    
  if t is not None:
    #plt.title(f"t = {t:03d}")
    pass

  if len(trajectories) >= 2:
    for i in range(1,len(trajectories)):
      coord_init, coord_last = trajectories[i-1][batch], trajectories[i][batch]
      for j in list_of_agents:
        x_in, y_in = coord_init[j, :]
        x_out, y_out = coord_last[j, :]
        a0.plot([x_in, x_out], [y_in, y_out], linestyle='dotted', c='gray')


  if utterances is not None:
    a1.imshow(torch.clone(utterances[batch]).detach().numpy(), cmap='cividis')
    a1.set_xticks([])
    a1.set_yticks([])



  
  #plt.axis("off")
  a0.set_xticks([])
  a0.set_yticks([])
  plt.gca().set_aspect('equal', adjustable='box')
  plt.gcf().set_size_inches(5, 6)
  plt.subplots_adjust(wspace=-1, hspace=-.25)
  if clear_previous:
    clear_output()
  
  if save:
    plt.savefig(f"{filename}{t:03d}.png",  bbox_inches='tight',pad_inches = 0.1)
  if return_plot:
    return f
  if show_plot:
    plt.show()
    
    
def visualize_world_and_vocab(locations, physical, args, 
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
                              show_plot=True):

  batch = batch
  player_icon = "o"
  landmark_icons = ["^", "s", "P", "X"]
  colors = ["r", "b", "g", "c", "m", "k"]

  n_entities = locations[batch].shape[0]

  world_dim = args["world_dim"] + tol

  f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

  if not show_first_quadrant:
    a0.set_xlim([-world_dim, world_dim])
    a0.set_ylim([-world_dim, world_dim])
  else:
    a0.set_xlim([0, world_dim])
    a0.set_ylim([0, world_dim])

  if n_agents is None:
    list_of_agents = [i for i in range(args["min_agents"])]
  else:
    list_of_agents = [i for i in range(n_agents)]


  a0.set_aspect(1)

  for i in reversed(range(n_entities)):
    color_id = int(torch.clone(physical[batch][i, 0]).detach().numpy())
    icon_id = int(torch.clone(physical[batch][i, 1]).detach().numpy())
    icon = player_icon if (i in list_of_agents) else landmark_icons[icon_id]
    color = colors[color_id]
    x, y = locations[batch][i,:].detach().numpy()
    if i in list_of_agents:
      a0.scatter(x, y, c=color, marker=icon, s=world_dim*20, alpha=0.4)
      a0.scatter(x, y, c=color, marker=icon, s=world_dim*5)
      if not show_first_quadrant:
        if show_speech_bubble and x < world_dim and y < world_dim and x > -world_dim and y > -world_dim:
            utter = np.argmax(torch.clone(utterances[batch][i]).detach().numpy())
            a0.text(x-world_dim/15, y-world_dim/15, f"[{utter:02d}]")
      else: 
        if show_speech_bubble and x < world_dim and y < world_dim and x > world_dim/15 and y > world_dim/15:
            utter = np.argmax(torch.clone(utterances[batch][i]).detach().numpy())
            a0.text(x-world_dim/15, y-world_dim/15, f"[{utter:02d}]")
    else:
      a0.scatter(x, y, c=color, marker=icon, s=world_dim*15)
    
  if t is not None:
    #plt.title(f"t = {t:03d}")
    pass

  if len(trajectories) >= 2:
    for i in range(1,len(trajectories)):
      coord_init, coord_last = trajectories[i-1][batch], trajectories[i][batch]
      for j in list_of_agents:
        x_in, y_in = coord_init[j, :]
        x_out, y_out = coord_last[j, :]
        a0.plot([x_in, x_out], [y_in, y_out], linestyle='dotted', c='gray')


  if utterances is not None:
    a1.imshow(torch.clone(utterances[batch]).detach().numpy(), cmap='cividis')
    a1.set_xticks([])
    a1.set_yticks([])



  
  #plt.axis("off")
  a0.set_xticks([])
  a0.set_yticks([])
  plt.gca().set_aspect('equal', adjustable='box')
  plt.gcf().set_size_inches(5, 6)
  plt.subplots_adjust(wspace=-1, hspace=-.25)
  if clear_previous:
    clear_output()
  
  if save:
    plt.savefig(f"{filename}{t:03d}.png",  bbox_inches='tight',pad_inches = 0.1)
  if return_plot:
    return f
  if show_plot:
    plt.show()
    
    
def visualize_episode_grid(timesteps, delay=0, clear_previous=True, track_trajectories=True, plot_utterances=True, title=None):

  
  for batch in range(8):

    filename = f"{batch}_"
    trajectories = []

    for t in range(args["n_timesteps"] + 1):

      if track_trajectories:
        trajectories.append(timesteps[t][0])

      locs, phys = timesteps[t][0], timesteps[t][1]
      if plot_utterances:
        utterances = timesteps[t][2]
        visualize_world_and_vocab(locs, phys, args, t=t, batch=batch, clear_previous=clear_previous, trajectories=trajectories, utterances=utterances, filename=filename, show_plot=False)

      time.sleep(delay)
  
  for t in range(args["n_timesteps"] + 1):
    fig, axs = plt.subplots(2, 4)
    row = 0
    col = 0
    for batch in range(8):
      
      img = plt.imread(f"{batch}_{t:03d}.png")
      
      axs[row , col].imshow(img)
      axs[row, col].axis('off')
      col += 1
      if col == 4:
        row = 1
        col = 0

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.gcf().set_size_inches(24, 12)
    plt.tight_layout()

    if title is not None:
      plt.title(title)
    plt.savefig(f"comp_{t:03d}.png")
    
    
