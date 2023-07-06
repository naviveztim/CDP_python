import sys
import numpy as np
import random
from Utils import utils
import pandas as pd

""" Implementation of Particle Swarm Optimization (PSO) algorithm- 
compares several candidate shapelets in a swarm. As result select the one which 
maximally separate two classes."""


class CircularListIterator:

    def __init__(self):
        self.float_list = [0.25, 0.71, 0.22, 0.12, 0.43, 0.03]
        self.index = -1

    def random(self) -> float:
        self.index = (self.index + 1) % len(self.float_list)
        return self.float_list[self.index]


class CandidateShapelet:

    def __init__(self
                 , length: int
                 , min_velocity: int, max_velocity: int
                 , min_position: int, max_position: int):
        self.optimal_split_distance: float = 0.0
        self.best_information_gain: float = sys.float_info.min
        self.length: int = length
        # TODO: Restore original random definitions
        self.position: np.array = np.zeros(length) #np.array([random.gauss(min_position, max_position) for _ in range(length)])
        self.velocity: np.array = np.zeros(length) #np.array([random.gauss(min_velocity, max_velocity) for _ in range(length)])
        self.best_position: np.array = self.position

    def copy(self, candidate):
        if isinstance(candidate, CandidateShapelet):
            self.optimal_split_distance = candidate.optimal_split_distance
            self.best_information_gain = candidate.best_information_gain
            self.length = candidate.length
            self.position[:len(candidate.position)] = candidate.position
            self.velocity[:len(candidate.velocity)] = candidate.velocity
            self.best_position = candidate.best_position

    '''
    def __repr__(self):
        print(f'optimal_split_distance: {self.optimal_split_distance}')
        print(f'best_information_gain: {self.best_information_gain}')
        print(f'length: {self.length}')
        print(f'position: {self.position}')
        print(f'velocity: {self.velocity}')
        print(f'best_position: {self.best_position}')
    '''

class ShapeletsPso:

    # PSO constants, according to: http://msdn.microsoft.com/en-us/magazine/hh335067.aspx
    W = 0.729  # inertia weight
    C1 = 1.49445  # cognitive / local weight
    C2 = 1.49445  # social / global weight

    # Stop optimization condition
    MAX_ITERATIONS = 20
    ITERATION_EPSILON = 0.0000001

    def __init__(self
                 , min_length: int, max_length: int
                 , step: int
                 , min_position: float, max_position: float
                 , min_velocity: float, max_velocity: float
                 , train_dataframe: pd.DataFrame):

        self.swarm: list = []
        self.min_position: float = min_position
        self.max_position: float = max_position
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity
        self.min_particle_length: int = min_length
        self.max_particle_length: int = max_length
        self.step: int = step
        self.train_dataframe: pd.DataFrame = train_dataframe
        #self.rand = CircularListIterator() # TEST # TODO: Restore real random vals

        # Init best particle
        self.best_particle: CandidateShapelet = CandidateShapelet(max_length
                                               , min_velocity, max_velocity
                                               , min_position, max_position)
        # Init swarm
        self._init_swarm()

    def _fitness_function(self, candidate: CandidateShapelet):

        """ Check the fitness of candidate shapelet"""

        distances = []
        # Calculate distances between candidate and given classes times series
        for _, time_series in self.train_dataframe.iterrows():

            distance_item = (time_series["class_index"]
                             , utils.subsequent_distance(time_series["values"], candidate.position))
            distances.append(distance_item)

        # Find the optimal split point given by candidate
        information_gain, split_point, optimal_entropy = \
            utils.calculate_information_gain(sorted(distances, key=lambda t: t[1]))

        #print(f'information_gain: {information_gain}'
        #      f', split_point: {split_point}'
        #      f', entropy: {optimal_entropy}')

        # Move candidate towards best position
        if candidate.best_information_gain < information_gain:
            candidate.best_information_gain = information_gain
            candidate.best_position = candidate.position
            candidate.optimal_split_distance = split_point

    def _init_swarm(self):

        for length in range(self.min_particle_length, self.max_particle_length+1, self.step):

            candidate = CandidateShapelet(length
                                          , self.min_velocity, self.max_velocity
                                          , self.min_position, self.max_position)
            # TODO: That is may be useless if random values introduced in __init__
            candidate.velocity = \
                np.array([(self.max_velocity - self.min_velocity)*random.random() + self.min_velocity for _ in
                         candidate.velocity])

            candidate.position = \
                np.array([(self.max_position - self.min_position) * random.random() + self.min_position for _ in
                         candidate.position])

            self._fitness_function(candidate)

            self.swarm.append(candidate)

            if candidate.best_information_gain > self.best_particle.best_information_gain:
                self.best_particle.copy(candidate)

    def start_pso(self):
        """ Find the best shapelet that distinguishes time series from two classes"""

        old_best_gain = 0.0
        new_best_gain = 0.0
        iteration = 0

        # TODO: Switch to ITERATION_EPSILON
        while True:
        #while iteration < ShapeletsPso.MAX_ITERATIONS:

            iteration += 1

            # Run competition between candidates
            for candidate in self.swarm:

                # Update candidate velocity
                for i in range(len(candidate.velocity)):

                    # TODO: Restore random vals
                    r1 = random.random()
                    r2 = random.random()
                    #r1 = self.rand.random()
                    #r2 = self.rand.random()

                    candidate.velocity[i] = self.W * candidate.velocity[i] + \
                        self.C1*r1*(candidate.best_position[i] - candidate.position[i]) + \
                        self.C2*r2*(self.best_particle.position[i] - candidate.position[i])

                # Update candidate position
                candidate.position += candidate.velocity

                # Check the fitness of current candidate
                self._fitness_function(candidate)

                # Update best particle
                if candidate.best_information_gain > self.best_particle.best_information_gain:
                    #self.best_particle = deepcopy(candidate)
                    self.best_particle.copy(candidate)

            old_best_gain = new_best_gain
            new_best_gain = self.best_particle.best_information_gain
            # TODO: Replace 'print' with 'logger.info'
            print(f'Iteration: {iteration}')
            print(f'Old best gain: {old_best_gain}')
            print(f'New best gain {new_best_gain}')


            if abs(old_best_gain - new_best_gain) <= ShapeletsPso.ITERATION_EPSILON:
                break
        self.best_particle.position = self.best_particle.position[:self.best_particle.length]
        self.best_particle.velocity = self.best_particle.velocity[:self.best_particle.length] # ??

        #print(f'optimal_split_distance: {self.best_particle.optimal_split_distance}')
        #print(f'best_information_gain: {self.best_particle.best_information_gain}')
        #print(f'length: {self.best_particle.length}')
        #print(f'position: {self.best_particle.position}')
        #print(f'velocity: {self.best_particle.velocity}')
        #print(f'best_position: {self.best_particle.best_position}')
