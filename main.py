import argparse
import os.path

from core.cdp import CDP
from utils.utils import from_ucr, to_ucr
from utils.logger import logger
import timeit


def get_arguments() -> argparse.Namespace:

    """ Parses command line arguments """

    parser = argparse.ArgumentParser(description='Time series classification, based on CDP method')
    parser.add_argument('-train', '--train', help='Specify csv file with train samples'
                        , required=False)
    parser.add_argument('-predict', '--predict', help='Specify csv file with samples to predict'
                        , required=False)
    parser.add_argument('-test', '--test', help='Specify csv file with test samples'
                        , required=False)
    parser.add_argument('-model_folder', '--model_folder', help='Specify folder where classifiers will be stored'
                        , required=False)
    parser.add_argument('-delimiter', '--delimiter', help='Delimiter used in dataset. Default: comma.'
                        , required=False, default=',')
    parser.add_argument('-compress', '--compress', help='Compression factor. Default: 1 (No compression)'
                        , required=False, default=1)
    parser.add_argument('-signal', '--signal', help='Use original signal or its derivative. Default: Original signal'
                        , required=False, default='s', choices=['s', 'd', 'S', 'D'])
    parser.add_argument('-normalize', '--normalize', help='Normalize the original signal? Default: Do not normalize'
                        , required=False, default='n', choices=['n', 'N', 'y', 'Y'])
    parser.add_argument('-nodes', '--nodes', help='Specify number of nodes in decision tree. Default: 2'
                        , required=False, default=2)
    parser.add_argument('-trees', '--trees', help='Specify number decision trees.'
                        , required=False, default=100)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        logger.info(str(e))
        parser.print_help()
        exit(1)

    return args


def show_arguments(args: argparse.Namespace):

    """ Displays picked arguments """
    logger.info(f'Train csv: {args.train}')
    logger.info(f'Model folder: {args.model_folder}')
    logger.info(f'Test csv: {args.test}')
    logger.info(f'Predict csv: {args.predict}')
    logger.info(f'Delimiter: {args.delimiter}')
    logger.info(f'Compress factor: {args.compress}')
    logger.info(f'Signal/Derivative?(S/D): {args.signal}')
    logger.info(f'Normalize?(Y/N): {args.normalize}')
    logger.info(f'Number of nodes in tree: {args.nodes}')
    logger.info(f'Number of trees in decision pattern: {args.trees}')


def main():
    # Get command line arguments
    args = get_arguments()

    # Display arguments
    show_arguments(args)

    # Obtain train dataset from csv file
    train_dataset = from_ucr(args.train, args.delimiter) if args.train else None

    # Initialize CDP
    cdp = CDP(dataset=train_dataset
              , model_folder=args.model_folder
              , num_classes_per_tree=int(args.nodes)
              , pattern_length=int(args.trees)
              , compression_factor=int(args.compress)
              , original_or_derivate=args.signal
              , normalize=args.normalize)

    # Train/Load the model
    if args.train:
        cdp.fit(args.model_folder)
    else:
        cdp.load_model(args.model_folder)

    if args.predict:

        # Obtain test dataset
        dataset = from_ucr(args.predict, delimiter=',', index=False)

        # Predict class indexes of a test dataset
        predicted_class_indexes = cdp.predict(dataset)

        # Format result filename
        original_filename = os.path.splitext(os.path.basename(args.predict))[0]
        directory_path = os.path.dirname(args.predict)
        output_filepath = os.path.join(directory_path, original_filename + '_predicted.csv')

        # Save results in UCR format
        to_ucr(dataset, predicted_class_indexes, output_filepath)

    # Test accuracy
    if args.test:

        # Obtain test dataset
        test_dataset = from_ucr(args.test, delimiter=',')

        # Predict class indexes of a test dataset
        predicted_class_indexes = cdp.predict(test_dataset)

        # Iterate through predicted indexes and check correspondence with the original
        num_correct_predictions = 0
        for i, row in test_dataset.iterrows():

            if row['class_index'] == predicted_class_indexes[i]:
                num_correct_predictions += 1

        logger.info(f"Accuracy: {100 * round(num_correct_predictions / len(predicted_class_indexes), 2)}%")


if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    logger.info(f"Execution time: {execution_time:.2f} seconds")









