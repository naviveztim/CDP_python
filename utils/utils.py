from collections import Counter
import numpy as np
from utils.dataset import Dataset
import numba
from numba import NumbaWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaWarning)


def try_except(func):
    """ Used to decorate functions with try/except clause """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"ERROR: {e}")
            result = None
            raise e
        return result
    return wrapper


@numba.jit()
def similarity_coefficient(s1: str, s2: str) -> float:
    """ Calculates similarity between two strings"""

    if len(s1) != len(s2):
        raise ValueError("The patterns have different length!")

    s1_array = np.array(list(s1))
    s2_array = np.array(list(s2))

    matches = np.sum(s1_array == s2_array)
    similarity = matches / len(s1)

    return similarity


@numba.jit()
def entropy(values: np.array) -> float:
    """ Calculates entropy of given sequence"""

    num_values = len(values)
    hist = Counter(values)

    values_array = np.array(list(hist.values()))
    probabilities = values_array / num_values

    entropy = -np.sum(probabilities * np.log(probabilities))

    return entropy


@numba.jit()
def assess_candidate_position(candidate_position: np.array
                              , train_dataframe: Dataset) -> tuple:

    """ Check the fitness of candidate shapelet"""

    distances = []

    for class_index, values in train_dataframe.iterrows():
        distance = subsequent_distance(values, candidate_position)
        distances.append((class_index, distance))

    distances_array = np.array(distances)
    sorted_distances = distances_array[distances_array[:, 1].argsort()]

    information_gain, split_point, optimal_entropy = calculate_information_gain(sorted_distances)

    return information_gain, split_point


@numba.jit()
def subsequent_distance(time_series_values: np.array
                        , candidate_position: np.array) -> float:
    """ Calculates the distance between two time series """

    min_distance = np.inf

    for i in range(len(time_series_values) - len(candidate_position) + 1):
        current_values = time_series_values[i:i + len(candidate_position)]
        squared_diff_sum = np.sum(np.square(current_values - candidate_position))
        min_distance = min(min_distance, squared_diff_sum)

    return min_distance


@numba.jit()
def calculate_information_gain(distances_array: np.array) -> tuple:
    """ Calculate information gain and optimal split distance from list of tuples in format
        (class_index, distance between the time series and the shapelet)"""

    information_gain = 0.0
    optimal_split_distance = 0.0
    optimal_entropy = -1.0

    #distances_array = np.array(distances)
    indexes = distances_array[:, 0]
    distances_values = distances_array[:, 1]

    I = entropy(indexes)

    prev_distance = distances_values[0]

    for i in range(1, len(distances_array)):
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


def process_dataset(dataset: Dataset
                    , compression_factor: int
                    , normalize: bool
                    , derivative: bool) -> Dataset:
    """ Process given dataframe with compression, normalization or extract derivative """

    if dataset is None or dataset.empty:
        return Dataset()

    dataset = dataset.copy()

    # Apply compression on time series
    if compression_factor > 1:
        dataset.apply_compression(compression_factor)

    # Extract derivative from time series
    if derivative:
        dataset.apply_derivative()

    # Apply normalization on time series
    if normalize:
        dataset.apply_normalization()

    return dataset

def to_ucr_format(dataframe: Dataset
                  , predicted_indexes: list
                  , filepath: str
                  , delimiter: str = ', '):
    """ Save pandas dataframe to csv file in UCR format"""

    raise NotImplemented
    '''
    dataframe.class_index = predicted_indexes

    # Convert list values to comma-separated strings
    dataframe.values = dataframe.values.apply(lambda x: delimiter.join(map(str, x)))
    comma_separated_string = ', '.join(str(x) for x in dataframe.values)

    # Save the DataFrame as a CSV file without column names
    pdf.to_csv(filepath, index=False, header=False, quoting=csv.QUOTE_NONE,  escapechar=' ')

    # Remove spaces in csv file
    for line in fileinput.input(filepath, inplace=True):
        sys.stdout.write(line.replace(' ', ''))
    '''






