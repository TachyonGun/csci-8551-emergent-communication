class GymConfig:
    def __init__(self, use_utterances: bool = False, penalize_words: bool = False, n_epochs: int = 1, learning_rate=None,
                 batch_size: int = 2, n_timesteps: int = 2, num_shapes: int = 4, num_colors: int = 4,
                 max_agents: int = 2, min_agents: int = 2, max_landmarks: int = 4, min_landmarks: int = 4,
                 vocab_size: int = 20, world_dim: int = 16, oov_prob=None, load_model_weights: str = None,
                 save_model_weights: str = None, use_cuda: bool = False, show_timestep: bool = True,
                 collect_state_history: bool = True, memory_size = 10, **kwargs):
        self.use_utterances = use_utterances
        self.penalize_words = penalize_words
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.num_shapes = num_shapes
        self.num_colors = num_colors
        self.max_agents = max_agents
        self.min_agents = min_agents
        self.max_landmarks = max_landmarks
        self.min_landmarks = min_landmarks
        self.vocab_size = vocab_size
        self.world_dim = world_dim
        self.oov_prob = oov_prob
        self.load_model_weights = load_model_weights
        self.save_model_weight = save_model_weights
        self.use_cuda = use_cuda
        self.show_timesteps = show_timestep
        self.collect_state_history = collect_state_history
        self.memory_size = 10

    def __str__(self):
        return f'''
                use_utterances: {self.use_utterances}
                penalize_words: {self.penalize_words}
                n_epochs: {self.n_epochs}
                learning_rate: {self.learning_rate}
                batch_size: {self.batch_size}
                n_timesteps: {self.n_timesteps}
                num_shapes: {self.num_shapes}
                num_colors: {self.num_colors}
                max_agents: {self.max_agents}
                min_agents: {self.min_agents}
                max_landmarks: {self.max_landmarks}
                min_landmarks: {self.min_landmarks}
                vocab_size: {self.vocab_size}
                world_dim: {self.world_dim}
                oov_prob: {self.oov_prob}
                load_model_weights: {self.load_model_weights}
                save_model_weights: {self.save_model_weight}
                use_cuda: {self.use_cuda}
                show_timestep: {self.show_timesteps}
        '''

