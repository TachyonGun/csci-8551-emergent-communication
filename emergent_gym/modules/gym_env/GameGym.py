from emergent_gym.modules.gym_env.Config.GameGymConfig import GymConfig
import gym
from gym import spaces
import torch
from torch.autograd import Variable


class GameGym(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config: GymConfig):
        super(GameGym, self).__init__()
        self.config = GymConfig

        self.collect_state_history = config.collect_state_history
        self.state_history = []
        self.batch_size = config.batch_size  # scalar: num games in this batch
        self.using_utterances = config.use_utterances  # bool: whether current batch allows utterances
        self.using_cuda = config.use_cuda
        self.num_agents = config.max_agents  # scalar: number of agents in this batch
        self.num_landmarks = config.max_landmarks  # scalar: number of landmarks in this batch
        self.num_entities = self.num_agents + self.num_landmarks  # type: int

        # Goto, and do nothing
        # TODO: Set this to 3 once gaze is implemented?
        self.action_space = spaces.Discrete(2)

        if self.using_cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        # Observable properties of agents
        locations = torch.rand(self.batch_size, self.num_entities, 2) * config.world_dim
        colors = (torch.rand(self.batch_size, self.num_entities, 1) * config.num_colors).floor()
        shapes = (torch.rand(self.batch_size, self.num_entities, 1) * config.num_shapes).floor()

        # Goal properties
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

        self.observation_space = {
            "utterances": self.utterances,
            "memories": self.memories,
            "locations": self.locations,
            "physical": self.physical,
            "goals": self.goals
        }

        new_obs = self.goals[:,:,:2] - agent_baselines

        # [batch_size, num_agents, 2] [batch_size, num_agents, 1]
        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)
        if self.collect_state_history:
            self.state_history.append(self.return_state())


    def return_state(self):
        return [self.locations, self.physical, self.utterances]

    def step(self, action):
        self.locations = self.locations + action['movements']
        agent_baselines = self.locations[:, :self.num_agents]
        self.observations = self.locations.unsqueeze(1) - agent_baselines.unsqueeze(2)
        new_obs = self.goals[:, :, :2] - agent_baselines
        goal_agents = self.goals[:, :, 2].unsqueeze(2)
        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)

        if self.using_utterances:
            self.utterances = action['utterances']

            # Update state history
            if self.collect_state_history:
                self.state_history.append(self.return_state())

            return self.compute_cost(action['movements'], action['goal_predictions'], action['utterances'])
        else:
            # Update state history
            if self.collect_state_history:
                self.state_history.append(self.return_state())

            return self.compute_cost(action['movements'], action['goal_predictions'])

    def reset(self):
        # Reset the state of the environment to an initial state
        pass

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # Port over code
        pass
