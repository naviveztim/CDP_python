import sys
import numpy as np
import random
from Utils import utils
import pandas as pd

""" Implementation of Particle Swarm Optimization (PSO) algorithm- 
compares several candidate shapelets in a swarm. As result select the one which 
maximally separate two classes."""


class CandidateShapelet:

    def __init__(self, length
                 , min_velocity, max_velocity
                 , min_position, max_position):
        self.optimal_split_distance = 0.0
        self.best_information_gain = sys.float_info.min
        self.length = length
        self.position = np.array([random.gauss(min_position, max_position) for _ in range(length)])
        self.velocity = np.array([random.gauss(min_velocity, max_velocity) for _ in range(length)])
        self.best_position = self.position

    def copy(self, candidate):
        if isinstance(candidate, CandidateShapelet):
            self.optimal_split_distance = candidate.optimal_split_distance
            self.best_information_gain = candidate.best_information_gain
            self.length = candidate.length
            self.position[:len(candidate.position)] = candidate.position
            self.velocity[:len(candidate.velocity)] = candidate.velocity
            self.best_position = candidate.best_position

    def __repr__(self):
        print(f'optimal_split_distance: {self.optimal_split_distance}')
        print(f'best_information_gain: {self.best_information_gain}')
        print(f'length: {self.length}')
        print(f'position: {self.position}')
        print(f'velocity: {self.velocity}')
        print(f'best_position: {self.best_position}')


class ShapeletsPso:

    # PSO constants, according to: http://msdn.microsoft.com/en-us/magazine/hh335067.aspx
    W = 0.729  # inertia weight
    C1 = 1.49445  # cognitive / local weight
    C2 = 1.49445  # social / global weight

    # Stop optimization condition
    MAX_ITERATIONS = 20
    ITERATION_EPSILON = 0.0000001

    def __init__(self, min_length: int, max_length: int, step: int
                 , min_position: float, max_position: float
                 , min_velocity: float, max_velocity: float
                 , train_dataframe: pd.DataFrame):

        self.swarm = []
        self.min_position = min_position
        self.max_position = max_position
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_particle_length = min_length
        self.max_particle_length = max_length
        self.step = step
        self.train_dataframe = train_dataframe #ShapeletsPso.norm(train_dataframe)
        '''
        time_series_dataframe: pd.DataFrame
        class_index:int     |   values:list
        ---------------------------------------
            0               |   [1.2, 0.2, 0.09, ...]
            0               |   [1.1, 0.9, 0.4, ....]
            1               |   [2, 2, 2, ...]
            1               |   [1.1, 1.9, 2.1, ...]
        '''

        # Init best particle
        self.best_particle = CandidateShapelet(max_length
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

            self._fitness_function(candidate)

            self.swarm.append(candidate)

            if candidate.best_information_gain > self.best_particle.best_information_gain:
                self.best_particle.copy(candidate)

    def start_pso(self):
        """ Find the best shapelet that distinguishes time series from two classes"""

        old_best_gain = 1.0
        new_best_gain = 0.0
        iteration = 0
        # TODO: Switch to ITERATION_EPSILON
        while abs(old_best_gain - new_best_gain) > ShapeletsPso.ITERATION_EPSILON:
        #while iteration < ShapeletsPso.MAX_ITERATIONS:

            # Run competition between candidates
            for candidate in self.swarm:

                # Update candidate velocity
                for i in range(len(candidate.velocity)):

                    r1 = random.random()
                    r2 = random.random()

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
            print(f'Old best gain: {old_best_gain}')
            print(f'New best gain {new_best_gain}')

            iteration += 1
            print(f'Iteration: {iteration}')
