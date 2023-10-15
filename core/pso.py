""" Particle Swarm Optimization (PSO) approach for finding
    shapelet that mostly separate two classes of time series"""
import sys
import numpy as np
from utils.utils import assess_candidate_position
from utils.dataset import Dataset
import numba
from numba import NumbaWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaWarning)


class CandidateShapelet:
    """ Shapelet candidate implementation"""
    def __init__(self
                 , length: int
                 , min_velocity: float, max_velocity: float
                 , min_position: float, max_position: float):
        self.optimal_split_distance: float = 0.0
        self.best_information_gain: float = sys.float_info.min
        self.length: int = length
        self.position: np.array = np.random.uniform(min_position, max_position, length)
        self.velocity: np.array = np.random.uniform(min_velocity, max_velocity, length)
        self.best_position: np.array = self.position

    def copy(self, candidate):
        """ Copy so found the best parameters to given candidate"""

        if isinstance(candidate, CandidateShapelet):
            self.optimal_split_distance = candidate.optimal_split_distance
            self.best_information_gain = candidate.best_information_gain
            self.length = candidate.length
            self.position[:len(candidate.position)] = candidate.position
            self.velocity[:len(candidate.velocity)] = candidate.velocity
            self.best_position = candidate.best_position


class ShapeletsPso:
    """ Implementation of Particle Swarm Optimization (PSO) algorithm
        compares several candidate shapelets in a swarm. As result select the one which
        maximally separate two classes."""

    # Constants of the process (empirically found)
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
                 , train_dataframe: Dataset
                 ):

        self.swarm: list = []
        self.min_position: float = min_position
        self.max_position: float = max_position
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity
        self.min_particle_length: int = min_length
        self.max_particle_length: int = max_length
        self.step: int = step
        self.train_dataframe = train_dataframe

        # Init best particle
        self.best_particle: CandidateShapelet = CandidateShapelet(max_length
                                               , min_velocity, max_velocity
                                               , min_position, max_position)
        # Init swarm
        self._init_swarm()

    @numba.jit()
    def _init_swarm(self):
        """ Initialize candidate shapelets' parameters """

        for length in range(self.min_particle_length, self.max_particle_length+1, self.step):

            # Create candidate shapelet
            candidate = CandidateShapelet(length
                                          , self.min_velocity, self.max_velocity
                                          , self.min_position, self.max_position)

            # Assess candidate position
            information_gain, split_point = \
                assess_candidate_position(candidate.position
                                          , self.train_dataframe)

            # Move candidate towards best position
            if candidate.best_information_gain < information_gain:
                candidate.best_information_gain = information_gain
                candidate.best_position = candidate.position
                candidate.optimal_split_distance = split_point
                candidate.length = length

            # Create swarm of candidates
            self.swarm.append(candidate)

            # Initialize best candidate
            if candidate.best_information_gain > self.best_particle.best_information_gain:
                self.best_particle.copy(candidate)

    @numba.jit()
    def start_pso(self):
        """ Find the best shapelet that distinguishes time series from two classes"""

        new_best_gain = 0.0
        iteration = 0

        while True:

            iteration += 1
            for candidate in self.swarm:

                # Update candidate velocity
                for i in range(candidate.length):

                    r_1 = np.random.rand()
                    r_2 = np.random.rand()

                    candidate.velocity[i] = self.W * candidate.velocity[i] + \
                        self.C1*r_1*(candidate.best_position[i] - candidate.position[i]) + \
                        self.C2*r_2*(self.best_particle.position[i] - candidate.position[i])

                # Update candidate position
                candidate.position += candidate.velocity

                # Check the fitness of current candidate
                information_gain, split_point = \
                    assess_candidate_position(candidate.position
                                              , self.train_dataframe)

                # Move candidate towards best position
                if candidate.best_information_gain < information_gain:
                    candidate.best_information_gain = information_gain
                    candidate.best_position = candidate.position
                    candidate.optimal_split_distance = split_point

                # Update best particle
                if candidate.best_information_gain > self.best_particle.best_information_gain:
                    self.best_particle.copy(candidate)

            old_best_gain = new_best_gain
            new_best_gain = self.best_particle.best_information_gain

            if abs(old_best_gain - new_best_gain) <= ShapeletsPso.ITERATION_EPSILON:
                break

        self.best_particle.position = self.best_particle.position[:self.best_particle.length]