"""Turn supervised mean action scores into a point-mass QR prior.

BC constrains the mean, not the return distribution. Repeating each action's
mean across quantiles removes unconstrained random spread while preserving
the action scores for every feature vector. Subsequent RL learns the spread.
"""
import torch


@torch.no_grad()
def initialize_point_mass(head):
    q, actions, features = head.quantile_count, head.action_count, head.feature_dim
    weight = head.advantage.weight.reshape(q, actions, features)
    bias = head.advantage.bias.reshape(q, actions)
    weight.copy_(weight.mean(0, keepdim=True).expand_as(weight).clone())
    bias.copy_(bias.mean(0, keepdim=True).expand_as(bias).clone())
    if head.value is not None:
        weight = head.value.weight
        bias = head.value.bias
        weight.copy_(weight.mean(0, keepdim=True).expand_as(weight).clone())
        bias.copy_(bias.mean().expand_as(bias).clone())
