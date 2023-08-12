import argparse
from core.cdp import CDP
from utils.utils import from_ucr_txt
from utils.logger import logger
import timeit


def get_arguments() -> argparse.Namespace:

    """ Parses command line arguments """

    parser = argparse.ArgumentParser(description='Time series classification, based on CDP method')
    parser.add_argument('-train_dir', '--train_dir', help='Specify folder with train dataset', required=False)
    parser.add_argument('-model_dir', '--model_dir', help='Specify folder where classifiers will be stored'
                        , required=True)
    parser.add_argument('-test_dir', '--test_dir', help='Specify folder with test dataset', required=False)
    parser.add_argument('-delimiter', '--delimiter', help='Delimiter used in dataset. Default: comma.'
                        , required=False, default=',')
    parser.add_argument('-compress', '--compress', help='Compression factor. Default: 1 (No compression)', required=False
                        , default=1)
    parser.add_argument('-signal', '--signal', help='Use original signal or its derivative. Default: Original signal'
                        , required=False, default='s', choices=['s', 'd', 'S', 'D'])
    parser.add_argument('-normalize', '--normalize', help='Normalize the original signal? Default: Do not normalize'
                        , required=False, default='n', choices=['n', 'N', 'y', 'Y'])
    parser.add_argument('-nodes', '--nodes', help='Specify number of nodes in decision tree. Default: 2'
                        , required=False, default=2)
    parser.add_argument('-trees', '--trees', help='Specify number decision trees.'
                        , required=True, default=100)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        logger.info(str(e))
        parser.print_help()
        exit(1)

    return args


def show_arguments(args: argparse.Namespace):

    """ Displays picked arguments """

    logger.info(f'Train folder: {args.train_dir}')
    logger.info(f'Model folder: {args.model_dir}')
    logger.info(f'Test folder: {args.test_dir}')
    logger.info(f'Delimiter: {args.delimiter}')
    logger.info(f'Compress factor: {args.compress}')
    logger.info(f'Signal/Derivative?(S/D): {args.signal}')
    logger.info(f'Normalize?(Y/N): {args.normalize}')
    logger.info(f'Number of nodes in tree: {args.nodes}')
    logger.info(f'Number of trees in decision pattern: {args.trees}')


def main():
    # Get command line arguments
    args = get_arguments()

    # Display entered arguments
    show_arguments(args)

    # Obtain train dataset
    train_dataset = from_ucr_txt(args.train_dir, args.delimiter) if args.train_dir else None

    # Initialize CDP
    cdp = CDP(dataset=train_dataset
              , classifiers_folder=args.model_dir
              , num_classes_per_tree=int(args.nodes)
              , pattern_length=int(args.trees)
              , compression_factor=int(args.compress)
              , original_or_derivate=args.signal
              , normalize=args.normalize)

    # Train/Load the model
    if args.train_dir:
        cdp.fit(args.model_dir)
    else:
        cdp.load_model(args.model_dir)

    # Test accuracy
    if args.test_dir:

        # Obtain test dataset
        test_dataset = from_ucr_txt(args.test_dir, delimiter=',')

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









