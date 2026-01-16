## Import Block ##
import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
import logging
from typing import Any, TypeVar, SupportsFloat

## Helpers
logger = logging.getLogger("MIDAS_logger")

# # Might use these but they seem to be more effort than it's worth right now
# T_co = TypeVar("SpaceT_co", covariant=True) # Generic type for ActType and ObsType
# ActType = T_co
# ObsType = T_co
# RenderFrame = Any # I have no idea what actual type RenderFrame is supposed to be


## Classes ##
class RLEnv(gym.Env):
    '''
    Gymnasium environment for use with sb3 to train an optimization model.

    This RL implementation aims to produce a loadable model, rather than a single solution. 
    The model can then be queried to produce a solution for a range of similar problems.

    Developed following the guide https://gymnasium.farama.org/introduction/create_custom_env/

    Required methods:
        reset
        step
        render
        close
    Required attributes
        action_space
        observation_space
    
    Additional methods:
        _get_obs:
        _parse_action_space:
        _parse_obs_space:
    '''
    def __init__(self):
        super().__init__()
        pass

    # Note: * without an identifier captures any positional args, forcing seed and options to be passed as keyword arguments.
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        pass

    # Note: Make a more robust type check instead of Any?
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    def render(self) -> Any | list[Any] | None:
        pass

    def close(self) -> None:
        pass

    def _get_obs(self) -> gym.spaces.Space:
        '''
        Converts state to the correct observation format.

        Since observation_space is highly dependent on user input and SB3's restriction,
        this non-public method will serve as a centralized helper method to make development easier.

        Suggested by the aforementioned Gym guide
        '''
        pass

    def _get_info(self) -> dict[str, Any]:
        '''
        Provides auixiliary debug information
        '''
        pass

    def _parse_action_space(self) -> gym.spaces.Space:
        '''
        Parses user input to an appropriate action_space. 
        
        The resulting action_space needs to obey SB3 restrictions (unless otherwise specified),
        user specifications, and be appropriate for the problem at hand.
        '''
        pass

    def _parse_obs_space(self) -> gym.spaces.Space:
        '''
        Parses user input to an appropriate observation_space. 
        
        The resulting observation_space needs to obey SB3 restrictions (unless otherwise specified),
        user specifications, and be appropriate for the problem at hand.

        Some training algorithms will need a selector in order to determine which output to currently process
        '''
        pass


class SB3Agent():
    '''
    Handles the learning of a model using training algorithms from SB3
    '''
    def __init__(self):
        pass