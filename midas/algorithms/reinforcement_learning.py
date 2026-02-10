## Import Block ##
import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import logging
from typing import Any, TypeVar, SupportsFloat

from itertools import repeat
from midas.utils import optimizer_tools as optools
from midas.utils import LWR_fuelcyclecost
from midas.utils import LWR_averageenrichment
from midas.utils import termination_criteria as TC

## Helpers
logger = logging.getLogger("MIDAS_logger")

# # Might use these but they seem to be more effort than it's worth right now
# T_co = TypeVar("SpaceT_co", covariant=True) # Generic type for ActType and ObsType
# ActType = T_co
# ObsType = T_co
# RenderFrame = Any # I have no idea what actual type RenderFrame is supposed to be


## Classes ##
class Reinforcement_Learning():
    '''
    Handles the implementation and learning of RL agents.

    Written by Bradley Maher. 02/08/2026
    '''
    def __init__(self, opts, eval_func):
        self.opts = opts
        self.eval_func = eval_func

        # These values should be set later
        self.population = None
        self.pool = None
    
    def reproduction(self, pop_list, current_generation):
        '''
        Returns a list of SB3Agent objects, with pop_list specified as the initial states
        '''
        agents = []

        for individual in pop_list:
            agents.append(SB3Agent(self.opts, individual, self.population, self.eval_func))
        
        return agents
    
    def set_population(self, population):
        self.population = population

    def set_pool(self, pool):
        self.pool = pool


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
    
    Written by Bradley Maher. 01/16/2026
    '''
    def __init__(self, opts, initial, population, eval_func):
        super().__init__()

        self.opts = opts
        self._initial = initial
        self._current = self._initial

        # Population object to hold archives
        self.population = population

        # Evaluation function for fitness calculations
        self.eval_func = eval_func

        # For testing with the IPWR database, I am hardcoding 6 discrete actions for each of the 8 assembly locations
        self.action_space = gym.spaces.MultiDiscrete(np.array([6]*8))

        # Observation is the current core, and each of the 3 objective values
        self.observation_space = gym.spaces.Dict({
            'current_chromosome': gym.spaces.MultiDiscrete(np.array([6]*8)),
            'PinPowerPeaking': gym.spaces.Box(low=0),
            'FDeltaH': gym.spaces.Box(low=0),
            'cycle_length': gym.spaces.Box(low=0)
        })

    # NOTE: * without an identifier captures any positional args, forcing seed and options to be passed as keyword arguments.
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        
        # Reset agent state to the specified initial state
        self._current = self._initial

        # Examine the chromosome for our chosen parameters
        self._update_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # Note: Make a more robust type check instead of Any?
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Action is currently hardcoded to be a list of fuel types. Set the new state to the inputted action
        # Update this agent based on the action just taken
        self._current = action
        
        # The chromosome needs to be examined to find our chosen paramters and fitness now
        self._update_state()

        observation = self._get_obs()
        info = self._get_info()

        # self.population.current[0] should now be updated after calling _update_state
        reward = self.population.current[0].fitness_value

        terminated = True # For now, one episode will be producing one core

        return observation, reward, terminated, False, info

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
        observation = {}

        with self.population.current[0] as soln:
            observation['current_chromosome'] = soln.chromosome
            observation['pinpowerpeaking'] = soln.parameters['pinpowerpeaking']['value']
            observation['fdeltah'] = soln.parameters['fdeltah']['value']
            observation['cycle_length'] = soln.parameters['cycle_length']['value']

        return observation

    def _update_state(self, action):
        '''
        Helper function to update the agent's state based on the action taken.
        '''
        self._current = action

        with self.population.current[0] as soln: # I really hope this WITH statement works as I imagined it would
            # Set the solution object's chromosome to this state
            gene_map = ['FA1', 'FA2', 'FA3', 'FA4', 'FA5', 'FA6']
            soln.chromosome = [gene_map[gene] for gene in self._current]
            inactive = False

            # See if the chromosome has already been tested. If not, run our chosen code to get it
            try: 
                soln_index = self.population.archive['solutions'].index(soln.chromosome)
                soln.fitness_value = self.population.archive['fitnesses'][soln_index]
                soln.parameters = self.population.archive['parameters'][soln_index]
                
                inactive = True

                logger.debug(f"Fitness value for solution '{soln.name}' will be taken from archive entry: {soln_index}.")
            except ValueError:
                # Chromosome is unique. We will calculate the fitness

                logger.info("Calculating fitness")
                ## Execute and parse objective/constraint values
                if not inactive: # This guard is not really needed
                    soln = self.eval_func(soln, self.input)
                    if 'cost_fuelcycle' in self.input.objectives.keys():
                        soln.parameters = LWR_fuelcyclecost.get_fuelcycle_cost(soln, self.input)
                    if 'av_fuelenrichment' in self.input.objectives.keys():
                        soln.parameters = LWR_averageenrichment.get_avfuelenrichment(soln, self.input)
                    
                    ## Calculate fitness from objective/constriant values
                    soln.fitness_value = optools.Fitness.calculate(soln.parameters)

                    logger.info("Done!")

    def _get_info(self) -> dict[str, Any]:
        '''
        Provides auxiliary debug information
        '''
        return {}

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

    Written by Bradley Maher. 01/16/2026
    '''
    def __init__(self, opts, initial, population, eval_func):
        self.opts = opts
        self.initial = initial
        self.population = population
        self.eval_func = eval_func

        self.env = self._build_env()

        policy = 'MultiInputPolicy'

        if self.opts.rl_algortihm == 'PPO':
            self.model = sb3.PPO(policy, self.env, verbose=1)
        else:
            raise ValueError('Specified SB3 Algorithm either invalid or not inplemented')

    def _build_env(self):
        env = RLEnv(self.opts, self.initial, self.population, self.eval_func)

        # Wrappers
        env = ShiftMultiWrapper(env)
        env = Monitor(env)

        check_env(env)

        return env

    def train(self):
        self.model.learn(total_timesteps=self.opts.learning_generations)

    def full_predict(self):
        env = self.model.get_env()
        observation, _ = env.reset()
        action, _ = self.model.predict(observation)
        env.step(action) # Pass the action to update the env's internal population object

        return env.population.current[0]
        
    # @staticmethod
    # def validate_otps():
    #     hyper_param_defaults = {
    #         'policy': 'MultiInputPolicy'
    #     }
    #     pass


class ShiftMultiWrapper(gym.Wrapper):
    '''Stole this from the SB3 docs, needed for SB3 algorithms to work with offset discrete action spaces '''
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.action_space = gym.spaces.MultiDiscrete(env.action_space.nvec, start=0)

    def step(self, action):
        return self.env.step(action + self.env.action_space.start)