## Import Block ##
import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
import sb3_contrib

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

import stable_baselines3.common.logger as sb3_logging

import logging
from typing import Any, Optional, Union, SupportsFloat, Callable
from copy import deepcopy
from math import pow, exp, log

from itertools import repeat
from midas.utils import optimizer_tools as optools
from midas.utils import LWR_fuelcyclecost
from midas.utils import LWR_averageenrichment
from midas.utils import termination_criteria as TC

import midas_data

## Helpers
logger = logging.getLogger("MIDAS_logger")

# # Might use these but they seem to be more effort than it's worth right now
# T_co = TypeVar("SpaceT_co", covariant=True) # Generic type for ActType and ObsType
# ActType = T_co
# ObsType = T_co
# RenderFrame = Any # I have no idea what actual type RenderFrame is supposed to be

# Parameter schedules
def param_schedule(initial_value: float, schedule: str = 'constant', final_value: float = 0, power: float = 1) -> Callable[[float], float]:
    '''
    Various learning schedules, 
    '''

    def constant_func(*args) -> float:
        return initial_value

    def linear_func(prog_remaining: float) -> float:
         return (initial_value - final_value) * prog_remaining + final_value

    def power_func(prog_remaining: float) -> float:
        return (initial_value - final_value) * pow(prog_remaining, power) + final_value

    def exp_func(prog_remaining: float) -> float:
        return final_value * exp(log(initial_value, final_value) * prog_remaining)

    if schedule == 'constant':
        return constant_func
    elif schedule == 'linear':
        return linear_func
    elif schedule == 'power':
        return power_func
    elif schedule == 'exp':
        return exp_func


## Classes ##
class Reinforcement_Learning():
    '''
    Handles the implementation and training of RL models.

    Written by Bradley Maher. 02/08/2026
    '''
    def __init__(self, input, eval_func):
        self.input = deepcopy(input)
        self.eval_func = eval_func

        # These values should be set later
        self.population = None
        self.generation = None
        self.initial = None
        self.pool = None
        self.model = None
        self.env = None
        self.optimizer = None

        # Update the learning rate to be a callable
        self.input.model_kwargs['learning_rate'] = param_schedule(**self.input.model_kwargs['learning_rate'])
        # Update clip range
        self.input.model_kwargs['clip_range'] = param_schedule(**self.input.model_kwargs['clip_range'])
    
    def reproduction(self, *args):
        '''
        Train for population_size steps, returning the list of generated solutions.

        This function follows the same proccess as those found in the optimizer main loop:
        reset population -> generate solutions -> find fitness

        TODO: Explore using RolloutBuffers? Would decouple generation/training.
        '''
        # Reset current population
        self.population.current = []

        print('begining training')
        # episode_count = self.input.population_size / self.input.markov_kwargs['steps_per_game']
        # callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=episode_count, verbose=1)

        self.model.learn(total_timesteps=self.input.population_size, reset_num_timesteps=False)
        print('done training')

        return self.population.current
    
    def set_population(self, population):
        self.population = population
    
    def set_generation(self, generation):
        self.generation = generation

    def set_pool(self, pool):
        self.pool = pool
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def build_model(self):
        '''
        Builds the RL model based on the user inputs.

        For now, only builds a SB3 based PPO or A2C model.
        '''
        self.env = self._build_env()

        policy = 'MultiInputPolicy' # TODO: allow this to be changed by the user?

        if self.input.model_load_path is not None:
            if self.input.rl_algorithm == 'PPO':
                self.model = sb3.PPO.load(self.input.model_load_path, self.env)
            elif self.input.rl_algorithm == 'MaskablePPO':
                self.model = sb3_contrib.MaskablePPO.load(self.input.model_load_path, self.env)
            elif self.input.rl_algorithm == 'A2C':
                self.model = sb3.A2C.load(self.input.model_load_path, self.env)
            else:
                raise ValueError('Specified SB3 Algorithm either invalid or not inplemented')
        else:
            if self.input.rl_algorithm == 'PPO':
                self.model = sb3.PPO(policy, self.env, verbose=1, **self.input.model_kwargs)
            elif self.input.rl_algorithm == 'MaskablePPO':
                self.model = sb3_contrib.MaskablePPO(policy, self.env, verbose=1, **self.input.model_kwargs)
            elif self.input.rl_algorithm == 'A2C':
                self.model = sb3.A2C(policy, self.env, verbose=1, **self.input.model_kwargs)
            else:
                raise ValueError('Specified SB3 Algorithm either invalid or not inplemented')

        # TODO: Fix the logger!
        # # MUST set the model to have a null logger. Having a logger will interfere with pickling
        # self.model.set_logger(sb3_logging.configure(None, []))

    def _build_env(self):
        env = RLEnv(self.input, self.initial, self.population, self.generation, self.eval_func, self.optimizer)

        # Wrappers
        env = ShiftMultiWrapper(env)
        # env = gym.wrappers.FlattenObservation(env)
        env = Monitor(env)
        # env = make_vec_env(env, n_envs=self.input.population_size)

        # check_env(env)

        return env

    def save_model(self):
        self.model.save(path=midas_data.__odir__ + '/' + self.input.model_save_path)

    # @staticmethod
    # def validate_otps():
    #     hyper_param_defaults = {
    #         'policy': 'MultiInputPolicy'
    #     }
    #     pass


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
    def __init__(self, input, initial, population, generation, eval_func, optimizer):
        super().__init__()

        self.input = input
        self.soln = None

        # How should the model output chosen assemblies, as a 'type' or by properties?
        # input.genome contains assembly maps
        # input.fa_options['fuel'] contains assembly properties
        if self.input.model_mode == 'ordinal':
            # Set a gene map to convert from ordinals to assembly names
            # This is just the assembly names in the order written in the input file
            # Some semblance of logic should be given to their order (e.g. increasing energy)
            self.gene_map = list(self.input.genome.keys())
        elif self.input.model_mode == 'property':
            # Convert from output properties to a specified assembly by name
            raise ValueError('Property output not supported yet')
        
        # Population object to hold archives
        self.population = population

        # Generation object
        self.generation = generation

        # Evaluation function for fitness calculations
        self.eval_func = eval_func

        # main optimizer object to steal the generate_solution function from and fitness
        self.optimizer = optimizer

        # For testing with the IPWR database, I am hardcoding 6 discrete actions for each of the 8 assembly locations
        self.action_space = self._parse_action_space() 

        # Observation constructed according to the specified inputs in yaml file
        self.observation_space = self._parse_obs_space()

        # If no starting initial chromosome was given, generate a random one.
        if initial is None:
            initial = self.action_space.sample()
            initial = [self.gene_map[gene] for gene in initial]
        self._initial = initial
        logger.debug(f"Initial genome set to: {self._initial}")

        self._current = self._initial

        # Example obsevation space:
        # self.observation_space = gym.spaces.Dict({
        #     # Previous core data
        #     'prev_chromosome': gym.spaces.MultiDiscrete(np.array([6]*8)),
        #     'pinpowerpeaking': gym.spaces.Box(low=0, high=np.inf, dtype=np.float64),
        #     'fdeltah': gym.spaces.Box(low=0, high=np.inf, dtype=np.float64),
        #     'cycle_length': gym.spaces.Box(low=0, high=np.inf, dtype=np.float64),

        #     # Fuel assmebly data
        #     'FA1_enrichment': gym.spaces.Box(low=0, high=1, dtype=np.float64),
        #     'FA1_gad_count': gym.spaces.Box(low=0, high=64, dtype=np.int8),
        #     'FA1_gad_loading': gym.spaces.Box(low=0, high=1, dtype=np.float64),
        #     'FA1_gad_enrichment': gym.spaces.Box(low=0, high=1, dtype=np.float64),
        #     'FA1_map': gym.spaces.MultiBinary()
        #     ...
        # })

        self.cur_step = 0
        self.cur_try = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        # Reset step and try counter to start of episode
        self.cur_step = 0
        self.cur_try = 0

        # Scramble FA gene map if enabled
        if self.input.markov_kwargs['scramble_FA']:
            self.np_random.shuffle(self.gene_map)

        # Reset agent state to the specified initial state
        self._current = [self.gene_map.index(gene) for gene in self._initial]
        
        # Also reset the current 
        self._update_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # Note: Make a more robust type check instead of Any?
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Increase the amount of steps taken this game by one
        self.cur_try += 1

        past = self._current

        # Action is currently hardcoded to be a list of fuel types. Set the new state to the inputted action
        # Update this agent based on the action just taken
        self._current = action
        
        # The chromosome needs to be examined to find our chosen paramters and fitness now
        chromosome_is_valid = optools.Gene_Validity_check.abortive_check(self.input, 
                                                                         list(self.input.genome.keys()), 
                                                                         self.input.genome, 
                                                                         [self.input.nrow, self.input.ncol, self.input.num_assemblies, self.input.symmetry, self.input.calculation_type], 
                                                                         [self.gene_map[gene] for gene in self._current])
        if chromosome_is_valid:
            logger.debug(f"CODE EVAL")

            self._update_state()

            self.population.current.append(self.soln)

            observation = self._get_obs()
            info = self._get_info()

            reward = self.soln.fitness_value # - 10 * (self.cur_try - 1)

            self.cur_step += 1
            self.cur_try = 0
        else:
            # Reset to previous success and report
            # No need to run _update_state(), as (theoretically) the soln object should still be the initial soln
            logger.warning(f'ILLEGAL CHROMOSOME, is the action mask set up properly?')
            self._current = past

            observation = self._get_obs()
            info = self._get_info()

            reward = self.input.markov_kwargs['failed_chromosome_reward']

            # if 'fitness' in observation.keys():
            #     observation['fitness'] += failed_chromosome_reward

            # Enough failed steps have occured that we should move on to prevent hanging.
            if self.cur_try >= self.input.markov_kwargs['chromosome_rety_steps']:
                self.cur_step += 1
                self.cur_try = 0
            

        terminated = True if self.cur_step >= self.input.markov_kwargs['steps_per_game'] else False

        return observation, reward, terminated, False, info

    def render(self) -> Union[Any, list[Any], None]:
        pass

    def close(self) -> None:
        pass

    def _get_obs(self) -> Union[gym.spaces.Space, dict]:
        '''
        Converts state to the correct observation format.

        Since observation_space is highly dependent on user input and SB3's restriction,
        this non-public method will serve as a centralized helper method to make development easier.

        Suggested by the aforementioned Gym guide
        '''
        observation = {}

        # Reconstruct observation based on the inputs specified. 
        # This is very similar to the logic found in the observation space setup.
        for category, keys in self.input.model_inputs.items():
            if category == 'prev_chromosome':
                observation[category] = self._current
            elif category == 'chromosome_info':
                for key in keys:
                    if key == 'fitness':
                        observation[key] = self.soln.fitness_value
                    else:
                        observation[key] = np.array([self.soln.parameters[key]['value']])
            elif category == 'gene_info':
                for id, assembly in enumerate(self.gene_map):
                    assembly_options = self.input.fa_options['fuel'][assembly]
                    for key in keys:
                        if key == 'map':
                            observation[f'FA{id}_{key}'] = self.input.genome[assembly]['map']
                        else:
                            observation[f'FA{id}_{key}'] = assembly_options[key]                            
            else:
                raise ValueError(f'Unknown RL input category {category}')

        logger.debug(observation)

        return observation

    def _update_state(self):
        '''
        Helper function to update the agent's state based on the action taken.
        '''
        chromosome = [self.gene_map[gene] for gene in self._current]

        # Built the next solution
        self.soln = self.optimizer.generate_solution(f'Gen_{self.generation.current}_Indv_{len(self.population.current)}', chromosome)
        
        # See if the chromosome has already been tested. If not, run our chosen code to get it
        try: 
            soln_index = self.population.archive['solutions'].index(self.soln.chromosome)
            self.soln.fitness_value = self.population.archive['fitnesses'][soln_index]
            self.soln.parameters = self.population.archive['parameters'][soln_index]

            logger.debug(f"Fitness value for solution '{self.soln.name}' will be taken from archive entry: {soln_index}.")
        except ValueError:
            # Chromosome is unique. We will calculate the fitness

            logger.debug("Calculating fitness")
            
            ## Execute and parse objective/constraint values
            self.soln = self.eval_func(self.soln, self.input)
            if 'cost_fuelcycle' in self.input.objectives.keys():
                self.soln.parameters = LWR_fuelcyclecost.get_fuelcycle_cost(self.soln, self.input)
            if 'av_fuelenrichment' in self.input.objectives.keys():
                self.soln.parameters = LWR_averageenrichment.get_avfuelenrichment(self.soln, self.input)

            ## Calculate fitness from objective/constriant values
            self.soln.fitness_value = self.optimizer.fitness.calculate(self.soln.parameters)

            logger.debug("Done!")

            # Add to archive
            self.population.archive['solutions'].append(self.soln.chromosome)
            self.population.archive['fitnesses'].append(self.soln.fitness_value)
            self.population.archive['parameters'].append(self.soln.parameters)

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
        if self.input.model_mode == 'ordinal':
            space_length = len(self.input.genome[next(iter(self.input.genome.keys()))]['map'])
            space_depth = len(self.input.genome.keys())
            return gym.spaces.MultiDiscrete([space_depth]*space_length)
        else:
            raise ValueError('Ordinal mode is the only one implemented currently.')

    def _parse_obs_space(self) -> gym.spaces.Space:
        '''
        Parses user input to an appropriate observation_space. 
        
        The resulting observation_space needs to obey SB3 restrictions (unless otherwise specified),
        user specifications, and be appropriate for the problem at hand.

        Some training algorithms will need a selector in order to determine which output to currently process
        '''
        obs_space = {}

        for category, keys in self.input.model_inputs.items():

            if category == 'prev_chromosome':
                obs_space[category] = deepcopy(self.action_space)
            elif category == 'chromosome_info':
                for tag, values in keys.items():
                    if values[0] == 'Box':
                        obs_space[tag] = gym.spaces.Box(**values[1])
                    elif values[0] == 'MultiDiscrete':
                        obs_space[tag] = gym.spaces.MultiDiscrete(**values[1])
                    elif values[0] == 'MultiBinary':
                        obs_space[tag] = gym.spaces.MultiBinary(**values[1])
                    else:
                        raise ValueError(f'Error on tag {tag}: space type {values[0]} not supported.')
            elif category == 'gene_info':
                for id in range(len(self.gene_map)):
                    for tag, values in keys.items():
                        if values[0] == 'Box':
                            obs_space[f'FA{id}_{tag}'] = gym.spaces.Box(**values[1])
                        elif values[0] == 'MultiDiscrete':
                            obs_space[f'FA{id}_{tag}'] = gym.spaces.MultiDiscrete(**values[1])
                        elif values[0] == 'MultiBinary':
                            obs_space[f'FA{id}_{tag}'] = gym.spaces.MultiBinary(**values[1])
                        else:
                            raise ValueError(f'Error on tag {tag}: space type {values[0]} not supported.')
            else:
                raise ValueError(f'Unknown RL input category {category}')

        logger.debug(f"Observation space set to: \n {obs_space}")

        return gym.spaces.Dict(obs_space)

    def action_masks(self) -> list[bool]:
        '''
        Returns a boolean list corresponding to if a gene is valid in the chromosome location.

        Only used when using MaskablePPO

        Written by Bradley Maher 01/09/2026
        '''
        chromosome_length = len(self.input.genome[next(iter(self.input.genome.keys()))]['map'])
        genome_depth = len(self.input.genome.keys())

        action_mask = []

        for loc in range(chromosome_length):
            for gene in range(genome_depth):
                action_mask.append(bool(self.input.genome[self.gene_map[gene]]['map'][loc]))

        logger.debug(f'Current action mask: f{action_mask}')
        return action_mask


class ShiftMultiWrapper(gym.Wrapper):
    '''Stole this from the SB3 docs, needed for SB3 algorithms to work with offset discrete action spaces.'''
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.action_space = gym.spaces.MultiDiscrete(env.action_space.nvec, start=np.zeros(env.action_space.nvec.shape, dtype=int))

    def step(self, action):
        return self.env.step(action + self.env.action_space.start)
