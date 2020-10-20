import torch
from torch import nn
from intervals import Interval
import numpy as np
from tqdm.notebook import tqdm
import random

from env import OsillatorGpu
from Model import Actor

class SwitchController(nn.Module):
    def __init__(self, base_model_paths, switch_path, device, soft_choice=False):
        super(SwitchController, self).__init__()
        self.base_models = []
        for base_model_path in base_model_paths:
            base_model = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
            base_model.load_state_dict(torch.load(base_model_path, map_location=device))
            base_model.eval()
            self.base_models.append(base_model)
        self.switch_model = DQN(2, 2).to(device)
        self.switch_model.load_state_dict(torch.load(switch_path, map_location=device))
        self.switch_model.eval()
        
        self.soft_choice = soft_choice
        
    def forward(self, state, soft_choice=None):
        if soft_choice is None:
            soft_choice = self.soft_choice
        if soft_choice:
            switch_soft_action = self.switch_model.softact(state)
            control_actions = torch.cat([base_model(state).view(1) for base_model in self.base_models])
            switch_action = (switch_soft_action * control_actions).sum()
        else:
            switch_hard_action = self.switch_model.act(state, act)
            switch_action = self.base_models[switch_hard_action](state)
        return switch_action

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(state)
            action  = q_value.max(0)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action
    
    def softact(self, state, epsilon=0):
        if random.random() > epsilon:
            q_value = self.forward(state)
            soft_action = nn.Softmax(dim=-1)(q_value)
        else:
            probs_unnorm = torch.rand(2)
            soft_action = probs_unnorm / probs_unnorm.sum()
        return soft_action

def mdist_to_boundary(state, x0_min=-2, x0_max=2, x1_min=-2, x1_max=2, return_pnt=False):
    # define distance to boundary
    # state is tensor of shape [bsize, 2]
    x0, x1 = state[:, 0], state[:, 1]
    mdists = torch.stack([
        torch.abs(x0 - x0_min), 
        torch.abs(x0 - x0_max),
        torch.abs(x1 - x1_min), 
        torch.abs(x1 - x1_max)
    ], dim=-1)
    mdist, mdist_idx = torch.min(mdists, dim=-1)
    if return_pnt:
        mpnts = [
            [x0_min, x1.cpu().data],
            [x0_max, x1.cpu().data],
            [x0.cpu().data, x1_min],
            [x0.cpu().data, x1_max]
        ]
        return torch.Tensor(mpnts[mdist_idx.cpu().data])
    else:
        return mdist

def init_falsify_gdbp_search(model, device, init_state=[1, 1], n_iters=100, step_size=0.1, return_seq=False):
    
    init_state = torch.Tensor(init_state)
    state_histories = []
    
    for _ in tqdm(range(n_iters)):
        
        env = OsillatorGpu(x0=init_state[0], x1=init_state[1])
        next_state = env.state # env.reset(*init_state)
        state_history = []
        
        for _ in range(100):
            state_tensor = next_state.to(device)
            state_history.append(state_tensor)
            
            control_action = model(state_tensor)
            next_state, reward, done = env.step(control_action)
            
            if done:
                # print('done')
                break
        
        # deal with gdbp
        dxj_dx0_0 = torch.autograd.grad(next_state[0], state_history[0], retain_graph=True)[0]
        dxj_dx0_1 = torch.autograd.grad(next_state[0], state_history[0], retain_graph=False)[0]

        dxj_dx0 = torch.stack([dxj_dx0_0, dxj_dx0_1], axis=0) # not sure axis is 0 or -1
        d_xj = mdist_to_boundary(next_state.view(-1, 2), return_pnt=True) - next_state
        d_x0 = torch.matmul(d_xj.T, dxj_dx0)
        d_x0_norm = d_x0 / torch.norm(d_x0)
        
        # update
        init_state = init_state + step_size * d_x0_norm.detach()
        # print(init_state)
                
        state_history_np = np.stack([s.detach().cpu().numpy() for s in state_history], axis=0)
        state_histories.append(state_history_np)
        
        if len(state_history) < 100:
            break
    
    if return_seq:
        return state_histories
    else:
        return state_histories[-1][0]