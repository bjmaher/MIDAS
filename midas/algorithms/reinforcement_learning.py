## Import Block ##
import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.logger as sb3_logging
import logging
from typing import Any, Optional, Union, SupportsFloat
from copy import deepcopy

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
    Handles the implementation and training of RL models.

    Written by Bradley Maher. 02/08/2026
    '''
    def __init__(self, input, eval_func):
        self.input = input
        self.eval_func = eval_func

        # These values should be set later
        self.population = None
        self.generation = None
        self.initial = None
        self.pool = None
        self.model = None
        self.env = None
        self.optimizer = None
    
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

        self.model.learn(total_timesteps=self.population.size, reset_num_timesteps=False)
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

        policy = 'MultiInputPolicy' # TODO: allow this to be changed by the user

        if self.input.rl_algorithm == 'PPO':
            self.model = sb3.PPO(policy, self.env, verbose=1, **self.input.model_kwargs)
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
        self._initial = initial
        self._current = self._initial
        self.soln = None

        # How should the model output chosen assemblies, as a 'type' or by properties?
        # input.genome contains assembly maps
        # input.fa_options['fuel'] contains assembly properties
        if self.input.model_mode == 'ordinal':
            # Set a gene map to convert from ordinals to assembly names
            # THis is just the assembly names in the order written in the input file
            self.gene_map = [key for key in self.input.genome.keys()]
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
        self.action_space = gym.spaces.MultiDiscrete(np.array([6]*8))

        # Observation constructed according to the specified inputs in yaml file
        self.observation_space = self._parse_obs_space()

        # If no starting initial chromosome was given, generate a random one.
        if self._initial is None:
            self._initial = self.action_space.sample()

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

        # TODO: Allow this to be adjusted
        self.steps_per_game = 10
        self.cur_step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        # Reset step counter to start of episode
        self.cur_step = 0

        # Reset agent state to the specified initial state
        self._current = self._initial

        # Also reset the current 
        self._update_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # Note: Make a more robust type check instead of Any?
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Increase the amount of steps taken this game by one
        self.cur_step += 1

        # Action is currently hardcoded to be a list of fuel types. Set the new state to the inputted action
        # Update this agent based on the action just taken
        self._current = action
        
        # The chromosome needs to be examined to find our chosen paramters and fitness now
        self._update_state()
        self.population.current.append(self.soln)

        observation = self._get_obs()
        info = self._get_info()

        reward = self.soln.fitness_value

        terminated = True if self.cur_step >= self.steps_per_game else False

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
            if category == 'prev_core':
                for key in keys:
                    if key == 'prev_chromosome':
                        observation[key] = self._current
                    elif key == 'pinpowerpeaking':
                        observation[key] = np.array([self.soln.parameters['pinpowerpeaking']['value']])
                    elif key == 'fdeltah':
                        observation[key] = np.array([self.soln.parameters['fdeltah']['value']])
                    elif key == 'cycle_length':
                        observation[key] = np.array([self.soln.parameters['cycle_length']['value']])
                    elif key == 'fitness':
                        observation[key] = self.soln.fitness_value
            elif category == 'fuel_assemblies':
                for assembly, assembly_options in self.input.fa_options['fuel'].items():
                    for key in keys:
                        if key == 'enrichment':
                            observation[f'{assembly}_{key}'] = assembly_options['enrichment']
                        elif key == 'gad_count':
                            observation[f'{assembly}_{key}'] = assembly_options['gad_count']
                        elif key == 'gad_loading':
                            observation[f'{assembly}_{key}'] = assembly_options['gad_loading']
                        elif key == 'gad_enrichment':
                            observation[f'{assembly}_{key}'] = assembly_options['gad_enrichment']
                        elif key == 'map':
                            observation[f'{assembly}_{key}'] = self.input.genome[assembly]['map']

        obs_space = {}

        for category, keys in self.input.model_inputs.items():

            if category == 'prev_chromosome':
                observation[category] = self._current
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
                for assembly in self.input.fa_options['fuel'].keys():
                    for tag, values in keys.items():
                        if values[0] == 'Box':
                            obs_space[f'{assembly}_{tag}'] = gym.spaces.Box(**values[1])
                        elif values[0] == 'MultiDiscrete':
                            obs_space[f'{assembly}_{tag}'] = gym.spaces.MultiDiscrete(**values[1])
                        elif values[0] == 'MultiBinary':
                            obs_space[f'{assembly}_{tag}'] = gym.spaces.MultiBinary(**values[1])
                        else:
                            raise ValueError(f'Error on tag {tag}: space type {values[0]} not supported.')
            else:
                raise ValueError(f'Unknown RL input category {category}')

        return observation

    def _update_state(self):
        '''
        Helper function to update the agent's state based on the action taken.
        '''
        chromosome = [self.gene_map[gene] for gene in self._current]

        # Built the next solution
        self.soln = self.optimizer.generate_solution(f'Gen_{self.generation.current}_Indv_{len(self.population.current)}', chromosome)
        
        inactive = False # Why is this here?

        # See if the chromosome has already been tested. If not, run our chosen code to get it
        try: 
            soln_index = self.population.archive['solutions'].index(self.soln.chromosome)
            self.soln.fitness_value = self.population.archive['fitnesses'][soln_index]
            self.soln.parameters = self.population.archive['parameters'][soln_index]
            
            inactive = True

            logger.debug(f"Fitness value for solution '{self.soln.name}' will be taken from archive entry: {soln_index}.")
        except ValueError:
            # Chromosome is unique. We will calculate the fitness

            logger.debug("Calculating fitness")
            ## Execute and parse objective/constraint values
            if not inactive: # This guard is not really needed

                self.soln = self.eval_func(self.soln, self.input)
                if 'cost_fuelcycle' in self.input.objectives.keys():
                    self.soln.parameters = LWR_fuelcyclecost.get_fuelcycle_cost(self.soln, self.input)
                if 'av_fuelenrichment' in self.input.objectives.keys():
                    self.soln.parameters = LWR_averageenrichment.get_avfuelenrichment(self.soln, self.input)

                ## Calculate fitness from objective/constriant values
                self.soln.fitness_value = self.optimizer.fitness.calculate(self.soln.parameters)
                logger.debug("Done!")

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
                for assembly in self.input.fa_options['fuel'].keys():
                    for tag, values in keys.items():
                        if values[0] == 'Box':
                            obs_space[f'{assembly}_{tag}'] = gym.spaces.Box(**values[1])
                        elif values[0] == 'MultiDiscrete':
                            obs_space[f'{assembly}_{tag}'] = gym.spaces.MultiDiscrete(**values[1])
                        elif values[0] == 'MultiBinary':
                            obs_space[f'{assembly}_{tag}'] = gym.spaces.MultiBinary(**values[1])
                        else:
                            raise ValueError(f'Error on tag {tag}: space type {values[0]} not supported.')
            else:
                raise ValueError(f'Unknown RL input category {category}')

        print(obs_space)

        return gym.spaces.Dict(obs_space)


class ShiftMultiWrapper(gym.Wrapper):
    '''Stole this from the SB3 docs, needed for SB3 algorithms to work with offset discrete action spaces.'''
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.action_space = gym.spaces.MultiDiscrete(env.action_space.nvec, start=np.zeros(env.action_space.nvec.shape, dtype=int))

    def step(self, action):
        return self.env.step(action + self.env.action_space.start)
