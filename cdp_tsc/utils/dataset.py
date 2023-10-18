import numpy as np
import csv
from cdp_tsc.utils.logger import logger


class Dataset:

    def __init__(self, filepath: str = None
                     , delimiter: str = None
                     , class_indexes: np.array = None
                     , values: np.array = None
                     , no_indexes=False):

        if filepath is not None and delimiter is not None:
            self.class_indexes, self.values = self._load_dataset(filepath, delimiter, no_indexes)
            self.empty = True if self.class_indexes.size == 0 or self.values.size == 0 else False
        elif class_indexes is not None and values is not None:
            self.class_indexes = class_indexes
            self.values = values
            self.empty = False
        else:
            self.class_indexes = None
            self.values = None
            self.empty = True

    def iterrows(self):
        for class_index, row in zip(self.class_indexes, self.values):
            yield class_index, row

    @staticmethod
    def _load_dataset(filepath: str, delimiter: str, no_indexes: bool) -> tuple:
        """Load a UCR type data set from a local folder. """

        try:
            # Parse data file
            data = np.genfromtxt(filepath, delimiter=delimiter)

            # Extract class indexes and values from UCR like txt file
            if no_indexes:
                # There are only values-no indexes in the file
                values = data[:, 0:]
                class_indexes = np.full(values.shape[0], -1)
            else:
                # Indexes are at the beginning of every row
                values = data[:, 1:]
                class_indexes = data[:, 0]
                class_indexes = class_indexes.astype('float64').astype('int64')

        except FileNotFoundError:
            logger.info("ERROR: File not found!")
            raise
        except OSError:
            logger.info("ERROR: OS error appeared!")
            raise
        except Exception as e:
            logger.info(f"ERROR: {str(e)}")
            raise

        return class_indexes, values

    def copy(self):
        """ Makes shallow copy of Dataset object """

        new_copy = Dataset(None
                           , None
                           , np.copy(self.class_indexes)
                           , np.copy(self.values)
                           )

        return new_copy

    def apply_normalization(self):

        """Apply normalization: (x - m)/sigma on given time series """

        # Calculate the mean and standard deviation for each row
        mean_values = np.mean(self.values, axis=1, keepdims=True)
        std_values = np.std(self.values, axis=1, keepdims=True)

        # Normalize each row separately using (x - mean) / std
        self.values = (self.values - mean_values) / std_values

    def apply_compression(self, compression_factor: int):
        """ Takes averaged sample of every 'compression_factor' samples """

        new_values = []
        for x in self.values:
            rolling_mean = np.convolve(x, np.ones(compression_factor) / compression_factor, 'valid')
            new_values.append(rolling_mean[::compression_factor])
        self.values = np.array(new_values)

    def apply_derivative(self):
        """ Takes derivatives of time series in given time series"""
        self.values = np.diff(self.values, axis=1)

    def filter_by_class(self, class_index):
        """ """

        filtered_indices = np.where(self.class_indexes == class_index)[0]
        return Dataset(None, None, self.class_indexes[filtered_indices], self.values[filtered_indices])

    def to_ucr_format(self, filename: str, delimiter: str = ','):
        """ Save predicted class indexes along with given time series"""

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file
                                , delimiter=delimiter
                                , quotechar='"'
                                , quoting=csv.QUOTE_MINIMAL)

            for class_index, row_values in zip(self.class_indexes, self.values):
                data_row = [class_index] + list(row_values)
                writer.writerow(data_row)
