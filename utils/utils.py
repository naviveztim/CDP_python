import numpy as np
import sys
from collections import Counter
from math import log
import pandas as pd
import csv
from utils.logger import logger


def try_except(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"An exception occurred: {e}")
            result = None
        return result
    return wrapper

def similarity_coeff(s1: str, s2: str) -> int:
    """ Find similarity coefficient between two strings with equal length"""
    if len(s1) != len(s2):
        raise Exception("The patterns have different length!")

    return sum(1 for x, y in zip(s1, s2) if x == y) / len(s1)


def entropy(values: list) -> float:

    """ Calculates entropy of given sequence"""
    num_values = len(values)

    hist = dict(Counter(values))

    d = dict((k, v/num_values) for k, v in hist.items())

    entropy = 0.0
    for k in d:
        entropy += -d[k]*log(d[k])

    return entropy


def subsequent_distance(time_series_values: np.array, candidate_position: np.array):

    """ Finds the minimum distance between candidate shapelet and given time series,
    Assume that candidate shapelet length is smaller that time series length. """

    # Iterate through time series and check the distance between every subsequent
    # chunk and candidate shapelet
    min_distance = sys.float_info.max
    for i in range(len(time_series_values) - len(candidate_position) + 1):
        current_values = np.array(time_series_values[i:i+len(candidate_position)])
        min_distance = min(min_distance, np.sum(np.square(current_values - candidate_position)))

    return min_distance


def calculate_information_gain(distances: list):

    """ Calculate information gain and optimal split distance from list of tuples in format
    (class_index, distance between time series and shapelet)"""
    information_gain = 0.0
    optimal_split_distance = 0.0
    optimal_entropy = -1.0

    # Find initial (without division) entropy
    indexes = [item[0] for item in distances]
    I = entropy(indexes)

    prev_distance = distances[0][1]
    for distance in distances:
        if distance[1] > prev_distance:
            d = prev_distance + (distance[1] - prev_distance) / 2.0
            prev_distance = distance[1]

            # Divide items based on split distance - d
            h1 = [item[0] for item in distances if item[1] <= d]
            h2 = [item[0] for item in distances if item[1] > d]

            # Find entropy of every part after division
            I1 = entropy(h1)
            I2 = entropy(h2)

            # Find fractions
            f1 = len(h1) / len(indexes)
            f2 = len(h2) / len(indexes)

            # Find information gain

            current_entropy = f1*I1 + f2*I2
            current_information_gain = I - current_entropy
            if current_information_gain > information_gain:
                information_gain = current_information_gain
                optimal_split_distance = d
                optimal_entropy = current_entropy

    return information_gain, optimal_split_distance, optimal_entropy


def from_ucr_txt(filepath: str, delimiter: str = ',') -> pd.DataFrame:

    """ UCR dataset format is a pure text format where every row starts
    with time series index and is followed by value of the time series. The lengths of those time series
    might not be the same"""

    if not filepath:
        return None

    pdf = pd.DataFrame(columns=['class_index', 'values'])
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            pdf = pdf.append({'class_index': int(row[0])
                              , 'values': [float(x) for x in row[1:]]}
                              , ignore_index=True)

    logger.debug(f'Number of train samples: {pdf.shape[0]}')
    return pdf






