import torch
import torch.nn as nn
from stable_baselines3 import PPO

class CompactEnsemble(nn.Module):
    def __init__(self, model1, model2, freeze_pretrained=True):
        super().__init__()
        
        # Extract and freeze pretrained MLPs
        self.mlp1 = model1.policy.mlp_extractor
        self.mlp2 = model2.policy.mlp_extractor
        
        if freeze_pretrained:
            for param in [*self.mlp1.parameters(), *self.mlp2.parameters()]:
                param.requires_grad = False
        
        # New trainable output layers (64*2 = 128 input features)
        self.action_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, obs):
        # Get features from both MLPs
        feat1_policy = self.mlp1.policy_net(obs)
        feat1_value = self.mlp1.value_net(obs)
        feat2_policy = self.mlp2.policy_net(obs)
        feat2_value = self.mlp2.value_net(obs)
        
        # Combine features
        policy_features = torch.cat([feat1_policy, feat2_policy], dim=-1)
        value_features = torch.cat([feat1_value, feat2_value], dim=-1)
        
        # Output
        actions = self.action_head(policy_features)
        values = self.value_head(value_features)
        
        return actions, values

# Integration with Stable-Baselines3 PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution

class EnsemblePolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, model1, model2, **kwargs):
        # Store models temporarily
        self._model1 = model1
        self._model2 = model2
        
        # Call parent init first
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    def _build_mlp_extractor(self):
        # Create ensemble here where we can properly add it as a submodule
        self.ensemble = CompactEnsemble(self._model1, self._model2, freeze_pretrained=True)
        
        # Create a proper dummy mlp_extractor that inherits from nn.Module
        class DummyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim_pi = 128  # Combined output from both models
                self.latent_dim_vf = 128
                # Add a dummy layer so .apply() has something to work with
                self.dummy = nn.Linear(1, 1)
        
        self.mlp_extractor = DummyMLP()
    
    def _build(self, lr_schedule):
        # Build base components first (this will create default action/value nets)
        super()._build(lr_schedule)
        
        # Now replace with our ensemble components
        self.action_net = self.ensemble.action_head
        self.value_net = self.ensemble.value_head
    
    def forward(self, obs, deterministic=False):
        # Extract features using the base features extractor
        features = self.extract_features(obs)
        
        # Get ensemble outputs
        actions, values = self.ensemble(features)
        
        # Create action distribution
        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(actions, self.log_std)
        else:
            distribution = self.action_dist.proba_distribution(actions)
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        action_logits, values = self.ensemble(features)
        
        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(action_logits, self.log_std)
        else:
            distribution = self.action_dist.proba_distribution(action_logits)
        
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def predict_values(self, obs):
        features = self.extract_features(obs)
        _, values = self.ensemble(features)
        return values

# Simple training function using SB3 PPO
def train_with_ppo(model1_path, model2_path, env, total_timesteps=100000):
    # Load pretrained models
    model1 = PPO.load(model1_path, env=env)
    model2 = PPO.load(model2_path, env=env)
    
    # Create PPO with ensemble policy
    ppo_model = PPO(
        EnsemblePolicy,
        env,
        policy_kwargs={'model1': model1, 'model2': model2},
        learning_rate=1e-4,
        verbose=1
    )
    
    # Train with PPO algorithm (includes proper parameter updates)
    ppo_model.learn(total_timesteps=total_timesteps)
    return ppo_model

# Usage:
# trained_ensemble = train_with_ppo("model1.zip", "model2.zip", env, 50000)
# trained_ensemble.save("ensemble_ppo_model")