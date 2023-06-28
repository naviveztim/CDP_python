import numpy as np
import argparse
from ShapeletDataMining.cdp import CDP
from logger import logger


def get_arguments() -> argparse.Namespace:
    """ Parses command line arguments """

    parser = argparse.ArgumentParser(description='Time series classification, based on CDP method')
    parser.add_argument('-train_dir', '--train_dir', help='Specify folder with train dataset', required=True)
    parser.add_argument('-model_dir', '--model_dir', help='Specify folder where classifiers will be stored'
                        , required=True)
    parser.add_argument('-test_dir', '--test_dir', help='Specify folder with test dataset', required=False)
    parser.add_argument('-delimiter', '--delimiter', help='Delimiter used in dataset. Default: comma.'
                        , required=False, default=',')
    parser.add_argument('-compress', '--compress', help='Compress factor. Default: 1', required=False
                        , default=1)
    parser.add_argument('-signal', '--signal', help='Use original signal or its derivative. Default: Original signal'
                        , required=False, default='s', choices=['s', 'd', 'S', 'D'])
    parser.add_argument('-normalize', '--normalize', help='Normalize or not the original signal. Default: Do not normalize'
                        , required=False, default='n', choices=['n', 'N', 'y', 'Y'])
    parser.add_argument('-nodes', '--nodes', help='Specify number of nodes in decision tree. Default: 2'
                        , required=False, default=2)
    parser.add_argument('-trees', '--trees', help='Specify number decision trees.'
                        , required=True, default=100)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(str(e))
        parser.print_help()
        exit(1)

    return args


def show_arguments(args: argparse.Namespace):

    print(f'Train folder: {args.train_dir}')
    print(f'Model folder: {args.model_dir}')
    print(f'Test folder: {args.test_dir}')
    print(f'Delimiter: {args.delimiter}')
    print(f'Compress factor: {args.compress}')
    print(f'Signal/Derivative?(S/D): {args.signal}')
    print(f'Normalize?(Y/N): {args.normalize}')
    print(f'Number of nodes in tree: {args.nodes}')
    print(f'Number of trees in decision pattern: {args.trees}')


if __name__ == '__main__':

    # Get command line arguments
    args = get_arguments()

    # Check arguments
    show_arguments(args)

    # Initialize Concatenated Decision Paths main class
    cdp = CDP(train_dataset_filepath=args.train_dir
              , classifiers_folder=args.model_dir
              , delimiter=args.delimiter
              , num_classes_per_tree=int(args.nodes)
              , pattern_length=int(args.trees)
              , compression_factor=int(args.compress)
              , original_or_derivate=args.signal
              , normalize=args.normalize)

    # Train the model given parameters during initialization
    cdp.fit()

    if args.test_dir:
        cdp.predict(test_dataset_filepath=args.test_dir, delimiter=',')
    else:
        logger.info('Training was successful.')






