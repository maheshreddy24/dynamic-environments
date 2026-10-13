"""MuJoCo reproductions of the dynamic-manipulation simulation scenarios from
"Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA Manipulation"
(Syed, Jakobsson, Hao, Ichnowski; CoRL 2026), Appendix C.1.

    import dynamic_envs
    env = dynamic_envs.make("ConveyorCup-v0")          # plain env
    env = gymnasium.make("DynamicEnvs/ConveyorCup-v0")  # via gymnasium
"""
from .registry import SCENARIOS, available, make, register_all

register_all()

__all__ = ["SCENARIOS", "available", "make", "register_all"]
