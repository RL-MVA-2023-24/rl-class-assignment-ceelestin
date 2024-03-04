from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import numpy as np
import os
import torch

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.state_mean = np.array([360_000, 7_750, 287, 33, 36_808, 55], dtype=np.float32)
        self.state_std = np.array([128_788, 14_435, 345, 25, 70_083, 32], dtype=np.float32)
        
        self.reward_mean = 45_344
        self.reward_std = 33_407

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            #device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            with torch.no_grad():
                observation = torch.Tensor(self.normalize_state(observation, self.state_mean, self.state_std)).unsqueeze(0).to(self.device)
                Qs = [model(observation) for model in self.models]
                Qs = [weight * q / q.mean() for q, weight in zip(Qs, self.weights)]
                Q = torch.stack(Qs).mean(0)
                return torch.argmax(Q).item()

    def save(self, path):
        self.path = path + "/model_ensemble.pt"
        torch.save(self.model.state_dict(), self.path)
        return 

    def load(self):
        self.device = torch.device('cpu')
        self.experiments = ['/../exp1.pt',
                            '/../exp2.pt'
                            ]
        self.weights = np.ones(len(self.experiments))
        self.paths = [os.getcwd() + exp for exp in self.experiments]
        self.models = [self.network(self.device) for exp in self.experiments]
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(self.paths[i], map_location=self.device))
            model.eval()

    def network(self, device):
        state_dim = 6
        n_action = 4
        nb_neurons = 256
        network = torch.nn.Sequential(torch.nn.Linear(state_dim, nb_neurons),
                          torch.nn.ReLU(),
                          torch.nn.Linear(nb_neurons, nb_neurons),
                          torch.nn.ReLU(), 
                          torch.nn.Linear(nb_neurons, nb_neurons),
                          torch.nn.ReLU(), 
                          torch.nn.Linear(nb_neurons, nb_neurons),
                          torch.nn.ReLU(), 
                          torch.nn.Linear(nb_neurons, nb_neurons),
                          torch.nn.ReLU(),
                          torch.nn.Linear(nb_neurons, n_action)).to(device)
        return network
    
    def normalize_state(self, state, means, std):
        out = (state - means) / std
        return out
    
    def normalize_reward(self, reward, mean, std):
        return (reward - mean) / std
    
if __name__ == "__main__":
    
    agent = ProjectAgent()
    