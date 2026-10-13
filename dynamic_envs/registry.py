"""Scenario registry: one entry per simulation scenario evaluated in the paper.

    Table 1  constant velocity      : ConveyorCup, BeamCup, PolePushCup, RollingBall
    Table 1  acceleration/decel.    : ConveyorCupAccel, BeamCupAccel, PolePushCupDecel, RollingBallIncline
    Table 2  speed sweep            : ConveyorCupSpeedSweep (pass speed=0.0 ... 0.40)
    Table 3  complex                : AirHockey, BallisticCatch, MultiObject, PlaceInBox, MultiBalls,
                                      Occlusion, MidFlight, LangMultiObject
    Tables 9-11 stress tests        : MultipleDeflection, OcclusionDeflection, PlinkoDrop
                                      (pass window_ms=100/200/400/800/1600)
"""
import importlib

import gymnasium as gym

# name -> (module:Class, default kwargs, family, paper reference)
SCENARIOS = {
    "ConveyorCup-v0": ("transport:ConveyorCup", {}, "constant_velocity", "Tab. 1 Conv."),
    "BeamCup-v0": ("transport:BeamCup", {}, "constant_velocity", "Tab. 1 Beam"),
    "PolePushCup-v0": ("transport:PolePushCup", {}, "constant_velocity", "Tab. 1 Pole"),
    "RollingBall-v0": ("rolling_ball:RollingBall", {}, "constant_velocity", "Tab. 1 Roll"),
    "ConveyorCupAccel-v0": ("transport:ConveyorCup", dict(profile="accel"), "acceleration", "Tab. 1 Conv."),
    "BeamCupAccel-v0": ("transport:BeamCup", dict(profile="accel"), "acceleration", "Tab. 1 Beam"),
    "PolePushCupDecel-v0": ("transport:PolePushCup", dict(profile="accel"), "acceleration", "Tab. 1 Pole"),
    "RollingBallIncline-v0": ("rolling_ball:RollingBall", dict(profile="incline"), "acceleration",
                              "Tab. 1 Roll (g.)"),
    "ConveyorCupSpeedSweep-v0": ("transport:ConveyorCup", dict(speed=0.20), "speed_sweep", "Tab. 2"),
    "AirHockey-v0": ("air_hockey:AirHockey", {}, "complex", "Tab. 3 AH"),
    "BallisticCatch-v0": ("catching:BallisticCatch", {}, "complex", "Tab. 3 BC"),
    "MultiObject-v0": ("multi_object:MultiObject", {}, "complex", "Tab. 3 MO"),
    "PlaceInBox-v0": ("multi_object:PlaceInBox", {}, "complex", "Tab. 3 PiB"),
    "MultiBalls-v0": ("catching:MultiBalls", {}, "complex", "Tab. 3 MB"),
    "Occlusion-v0": ("catching:Occlusion", {}, "complex", "Tab. 3 Occ"),
    "MidFlight-v0": ("catching:MidFlight", {}, "complex", "Tab. 3 Mid"),
    "LangMultiObject-v0": ("multi_object:LangMultiObject", {}, "complex", "Tab. 3 L+MO"),
    "MultipleDeflection-v0": ("catching:MultipleDeflection", {}, "stress", "Tab. 9"),
    "OcclusionDeflection-v0": ("catching:OcclusionDeflection", {}, "stress", "Tab. 10"),
    "PlinkoDrop-v0": ("catching:PlinkoDrop", {}, "stress", "Tab. 11"),
}

NAMESPACE = "DynamicEnvs"


def _resolve(target):
    mod, cls = target.split(":")
    return getattr(importlib.import_module(f"dynamic_envs.tasks.{mod}"), cls)


def available():
    out = []
    for name, (target, _, _, _) in SCENARIOS.items():
        try:
            _resolve(target)
            out.append(name)
        except (ImportError, AttributeError):
            pass
    return out


def make(name, **kwargs):
    """Instantiate a scenario directly (no gymnasium wrappers)."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {list(SCENARIOS)}")
    target, defaults, family, _ = SCENARIOS[name]
    env_kwargs = dict(defaults)
    env_kwargs.update(kwargs)
    env = _resolve(target)(**env_kwargs)
    env.family = family
    env.scenario = name
    return env


def _entry(name):
    def factory(**kwargs):
        return make(name, **kwargs)
    return factory


def register_all():
    for name in SCENARIOS:
        env_id = f"{NAMESPACE}/{name}"
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=_entry(name), disable_env_checker=True)
