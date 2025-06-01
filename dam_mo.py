import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


register(
    id='Dam_MO',
    entry_point='dam_mo:Dam',
    kwargs={'nO': [0]}
)

class Dam(gym.Env):
    S = 1.0 # Reservoir surface
    W_IRR = 50. # Water demand
    H_FLO_U = 100. # Flooding threshold (upstream, i.e. height of dam) 50
    S_MIN_REL = 100. # Release threshold (i.e. max capacity)
    DAM_INFLOW_MEAN = 40. # Random inflow (e.g. rain)
    DAM_INFLOW_STD = 10.
    Q_MEF = 0.
    GAMMA_H2O = 1000. # water density
    W_HYD = 4.36 # Hydroelectric demand
    Q_FLO_D = 30. # Flooding threshold (downstream, i.e. releasing too much water)
    ETA = 1. # Turbine efficiency
    G = 9.81 # Gravity

    s_init = [9.6855361e+01, 
              5.8046026e+01, 
              1.1615767e+02, 
              2.0164311e+01, 
              7.9191000e+01, 
              1.4013098e+02, 
              1.3101816e+02,
              4.4351321e+01,
              1.3185943e+01,
              7.3508622e+01,]
    
    def __init__(self, nO=[0], penalize=False):
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,))
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.nO = nO
        self.penalize = penalize
        self.random_steps = 50000
        self.step_count = 0
        self.reward_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))

    def reset(self, seed=None, options=None):
        if not self.penalize:
            state = np.random.choice(Dam.s_init, size=1)
        else:
            state = np.random.randint(0, 160, size=1)
        
        self.state = state
        return self.state, {}

    def step(self, action):

        self.step_count += 1
        # bound the action
        actionLB = np.clip(self.state - Dam.S_MIN_REL, 0, None)
        actionUB = self.state

        # Penalty proportional to the violation
        if self.step_count < self.random_steps:
            action = self.action_space.sample()
        bounded_action = action * (actionUB - actionLB) + actionLB
        # print(f"action: {action}, bounded_action: {bounded_action}")
        # penalty = -self.penalize*np.abs(bounded_action - action)

        # transition dynamic
        action = bounded_action
        dam_inflow = np.random.normal(Dam.DAM_INFLOW_MEAN, Dam.DAM_INFLOW_STD, len(self.state))
        # small chance dam_inflow < 0
        n_state = np.clip(self.state + dam_inflow - action, 0, None)

        # cost due to excess level wrt a flooding threshold (upstream)
        r0 = -np.clip(n_state/Dam.S - Dam.H_FLO_U, 0, None)
        # deficit in the water supply wrt the water demand
        r1 = -np.clip(Dam.W_IRR - action, 0, None)
        
        q = np.clip(action - Dam.Q_MEF, 0, None)
        p_hyd = Dam.ETA * Dam.G * Dam.GAMMA_H2O * n_state / Dam.S * q / 3.6e6

        # deficit in hydroelectric supply wrt hydroelectric demand
        r2 = -np.clip(Dam.W_HYD - p_hyd, 0, None)
        # cost due to excess level wrt a flooding threshold (downstream)
        r3 = -np.clip(action - Dam.Q_FLO_D, 0, None)

        reward_list = [r0, r1, r2, r3]
        reward = np.sum(reward_list[r_index] for r_index in self.nO)

        self.state = n_state
        return n_state, reward, False, False, {}
