import numpy as np
import sys
from collections import Counter
import pandas as pd
from utils.logger import logger
import csv
import fileinput


def individual_use_only(func):
    """ Individual usage only - message generator """
    def wrapper(*args, **kwargs):
        logger.info("Warning: This software is free for individual use only.")
        logger.info("For more information please contact: cdp_project@outlook.com")
        return func(*args, **kwargs)
    return wrapper


def try_except(func):
    """ Used to decorate functions with try/except clause """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"An exception occurred: {e}")
            result = None
        return result
    return wrapper


def similarity_coefficient(s1: str, s2: str) -> float:
    """ Calculates similarity between two strings"""

    if len(s1) != len(s2):
        raise ValueError("The patterns have different length!")

    s1_array = np.array(list(s1))
    s2_array = np.array(list(s2))

    matches = np.sum(s1_array == s2_array)
    similarity = matches / len(s1)

    return similarity


def entropy(values: list) -> float:
    """ Calculates entropy of given sequence"""

    num_values = len(values)
    hist = Counter(values)

    values_array = np.array(list(hist.values()))
    probabilities = values_array / num_values

    entropy = -np.sum(probabilities * np.log(probabilities))

    return entropy


def assess_candidate_position(candidate_position: list, train_dataframe: list) -> tuple:
    """ Check the fitness of candidate shapelet"""

    distances = []

    for record in train_dataframe:
        class_index = record["class_index"]
        values = record["values"]

        distance = subsequent_distance(values, candidate_position)
        distances.append((class_index, distance))

    distances_array = np.array(distances)
    sorted_distances = distances_array[distances_array[:, 1].argsort()]

    information_gain, split_point, optimal_entropy = calculate_information_gain(sorted_distances)

    return information_gain, split_point


def subsequent_distance(time_series_values: list, candidate_position: list) -> float:
    """ Calculates the distance between two time series """

    time_series_array = np.array(time_series_values)
    candidate_array = np.array(candidate_position)

    min_distance = np.inf

    for i in range(len(time_series_values) - len(candidate_position) + 1):
        current_values = time_series_array[i:i + len(candidate_position)]
        squared_diff_sum = np.sum(np.square(current_values - candidate_array))
        min_distance = min(min_distance, squared_diff_sum)

    return min_distance


def calculate_information_gain(distances: list) -> tuple:
    """ Calculate information gain and optimal split distance from list of tuples in format
        (class_index, distance between the time series and the shapelet)"""

    information_gain = 0.0
    optimal_split_distance = 0.0
    optimal_entropy = -1.0

    distances_array = np.array(distances)
    indexes = distances_array[:, 0]
    distances_values = distances_array[:, 1]

    I = entropy(indexes)

    prev_distance = distances_values[0]

    for i in range(1, len(distances)):
        distance = distances_values[i]
        if distance > prev_distance:
            d = (prev_distance + distance) / 2.0
            prev_distance = distance

            h1_indexes = indexes[:i][distances_values[:i] <= d]
            h2_indexes = indexes[i:][distances_values[i:] > d]

            I1 = entropy(h1_indexes)
            I2 = entropy(h2_indexes)

            f1 = len(h1_indexes) / len(indexes)
            f2 = len(h2_indexes) / len(indexes)

            current_entropy = f1 * I1 + f2 * I2
            current_information_gain = I - current_entropy

            if current_information_gain > information_gain:
                information_gain = current_information_gain
                optimal_split_distance = d
                optimal_entropy = current_entropy

    return information_gain, optimal_split_distance, optimal_entropy


def to_ucr_format(pdf: pd.DataFrame, predicted_indexes: list, filepath: str, delimiter: str = ', '):
    """ Save pandas dataframe to csv file in UCR format"""

    pdf['class_index'] = predicted_indexes

    # Convert list values to comma-separated strings
    pdf['values'] = pdf['values'].apply(lambda x: delimiter.join(map(str, x)))

    # Save the DataFrame as a CSV file without column names
    pdf.to_csv(filepath, index=False, header=False, quoting=csv.QUOTE_NONE,  escapechar=' ')

    # Remove spaces in csv file
    for line in fileinput.input(filepath, inplace=True):
        sys.stdout.write(line.replace(' ', ''))


def from_ucr_format(filepath: str, delimiter: str = ',', index=True) -> pd.DataFrame:

    """ Reads data from csv file in UCR format and return it as pandas dataset, where the column
    'index' represents the time series class index, and column 'value' represents corresponding time series.
    UCR dataset format is a pure text format where every row starts
    with time series index and is followed by value of the time series. The lengths of those time series
    might not be the same"""

    if not filepath:
        return None

    pdf = pd.DataFrame(columns=['class_index', 'values'])
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            class_index = int(row[0]) if index else -1
            start_index = 1 if index else 0
            pdf = pdf.append({'class_index': class_index
                              , 'values': [float(x) for x in row[start_index:]]}, ignore_index=True)

    logger.debug(f'Number of samples: {pdf.shape[0]}')
    return pdf






