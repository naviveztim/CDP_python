""" Parse input arguments and apply train/predict/evaluate procedure on given dataset. """

import argparse
import os.path
import timeit
import sys
import numpy as np

from core.cdp import CDP
from utils.logger import logger
from utils.dataset import Dataset
from utils.utils import process_dataset


def get_arguments() -> argparse.Namespace:

    """ Parses command line arguments """

    parser = argparse.ArgumentParser(description='Time series classification, based on CDP method')
    parser.add_argument('-train', '--train', help='Specify csv file with train samples'
                        , required=False)
    parser.add_argument('-predict', '--predict'
                        , help='Specify csv file with samples to predict'
                        , required=False)
    parser.add_argument('-test', '--test'
                        , help='Specify csv file with test samples'
                        , required=False)
    parser.add_argument('-model_folder', '--model_folder'
                        , help='Specify folder where classifiers will be stored'
                        , required=False)
    parser.add_argument('-delimiter', '--delimiter'
                        , help='Delimiter used in dataset. Default: comma.'
                        , required=False, default=',')
    parser.add_argument('-compress', '--compress'
                        , help='Compression factor. Default: 1 (No compression)'
                        , required=False, default=1)
    parser.add_argument('-derivative', '--derivative'
                        , help='Use original signal or its derivative. Default: Original signal'
                        , required=False, action='store_true')
    parser.add_argument('-normalize', '--normalize'
                        , help='Normalize the original signal? Default: Do not normalize'
                        , required=False, action='store_true')
    parser.add_argument('-nodes', '--nodes'
                        , help='Specify number of nodes in decision tree. Default: 2'
                        , required=False, default=2)
    parser.add_argument('-trees', '--trees'
                        , help='Specify number decision trees.'
                        , required=False, default=100)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_error:
        logger.info(str(arg_error))
        parser.print_help()
        sys.exit(1)

    return args


def show_arguments(args: argparse.Namespace):
    """ Displays picked arguments """

    logger.info(f'Train csv: {args.train}')
    logger.info(f'Model folder: {args.model_folder}')
    logger.info(f'Test csv: {args.test}')
    logger.info(f'Predict csv: {args.predict}')
    logger.info(f'Delimiter: "{args.delimiter}"')
    logger.info(f'Compress factor: {args.compress}')
    logger.info(f'Derivative: {args.derivative}')
    logger.info(f'Normalize: {args.normalize}')
    logger.info(f'Number of nodes in tree: {args.nodes}')
    logger.info(f'Number of trees in decision pattern: {args.trees}')


def main():

    """ Main process function- train and evaluate prediction on given dataset"""
    np.random.seed(42)

    # Get command line arguments
    args = get_arguments()

    # Display arguments
    show_arguments(args)

    # Obtain train dataset from csv file
    train_dataset = Dataset(filepath=args.train
                            , delimiter=args.delimiter)

    # Apply pre-processing, defined by input parameters
    train_dataset = process_dataset(train_dataset
                                    , compression_factor=int(args.compress)
                                    , normalize=args.normalize
                                    , derivative=args.derivative)

    # Initialize CDP
    cdp = CDP(dataset=train_dataset
              , model_folder=args.model_folder
              , num_classes_per_tree=int(args.nodes)
              , pattern_length=int(args.trees)
              )

    # Train/Load the model
    if args.train:
        cdp.fit()
    else:
        cdp.load_model()

    if args.predict:

        # Obtain test dataset
        dataset = Dataset(filepath=args.predict
                          , delimiter=','
                          , no_indexes=True)

        # Apply pre-processing, already applied to train dataset
        dataset = process_dataset(dataset
                                  , compression_factor=int(args.compress)
                                  , normalize=args.normalize
                                  , derivative=args.derivative
                                  )

        # Predict class indexes of a test dataset
        dataset.class_indexes = cdp.predict(dataset)

        # Format result filename
        original_filename = os.path.splitext(os.path.basename(args.predict))[0]
        directory_path = os.path.dirname(args.predict)
        output_filepath = os.path.join(directory_path, original_filename + '_predicted.csv')

        # Save results in UCR format
        dataset.to_ucr_format(output_filepath)

    # Test accuracy
    if args.test:

        # Obtain test dataset
        test_dataset = Dataset(args.test, delimiter=',')

        # Apply pre-processing, already applied to train dataset
        test_dataset = process_dataset(test_dataset
                                       , compression_factor=int(args.compress)
                                       , normalize=args.normalize
                                       , derivative=args.derivative
                                       )

        # Predict class indexes of a test dataset
        predicted_class_indexes = cdp.predict(test_dataset)

        # Iterate through predicted indexes and check correspondence with the original
        num_correct_predictions = 0
        i = 0
        for index, row in test_dataset.iterrows():
            if index == predicted_class_indexes[i]:
                num_correct_predictions += 1
            i += 1

        average_accuracy = 100 * round(num_correct_predictions / len(predicted_class_indexes), 4)
        logger.info(f"Accuracy: {average_accuracy}%")


if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    logger.info(f"Execution time: {execution_time:.2f} seconds")
