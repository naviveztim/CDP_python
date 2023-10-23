## Python implementation of Concatenation Decision Paths (CDP)- fast and accurate method for time series classification 

### Overview 
Python implementation of the CDP algorithm posses following advantages: 
- **very fast** to (re)train (training time vary from seconds to minutes for datasets from [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/))
- produces **compact (~KB) models**, in comparison with large standard models (~100MB)  
- maintains **high accuracy** and is comparable or in some cases even more accurate than state-of-the-art algorithms (Fig.1) 
- python implementation does not depend on other machine learning package. It has only dependencies on standard python packages
- **very simple** to maintain (consists of 8 python files, spread in two folders)

### Installation 
pip install cdp-ts

### Training & Testing 

<pre>
from cdp_tsc.core.cdp import CDP
from cdp_tsc.utils.logger import logger
from cdp_tsc.utils.dataset import Dataset
from cdp_tsc.utils.utils import process_dataset
import numpy as np
from functools import wraps

TRAIN_DATASET_PATH = <<train filepath>>
TEST_DATASET_PATH = <<test filepath>>
DELIMITER = "\t"
MODELS_FOLDER_PATH = <<model folder path>>
COMPRESSION_FACTOR = 1,2,3,4...
NORMALIZE = True/False
DERIVATIVE = True/False
NUM_CLASSES_PER_TREE = 2
NUM_TREES = <<some number of trees>>


def train():
    """ Demo function that shows creating and training of CDP model"""
s
    # Obtain train dataset from 'ucr' type csv file
    train_dataset = Dataset(filepath=TRAIN_DATASET_PATH
                            , delimiter=DELIMITER)

    # Apply pre-processing
    train_dataset = process_dataset(dataset=train_dataset
                                    , compression_factor=COMPRESSION_FACTOR
                                    , normalize=NORMALIZE
                                    , derivative=DERIVATIVE)

    # Initialize CDP
    cdp = CDP(model_folder=MODELS_FOLDER_PATH
              , num_classes_per_tree=NUM_CLASSES_PER_TREE
              , num_trees=NUM_TREES
              )

    # Train the model
    cdp.fit(train_dataset)


def test():

    # Initialize CDP
    cdp2 = CDP(model_folder=MODELS_FOLDER_PATH
               , num_classes_per_tree=NUM_CLASSES_PER_TREE
               , num_trees=NUM_TREES
               )

    # Load already trained model 
    cdp2.load_model()

    # Obtain test dataset
    test_dataset = Dataset(filepath=TEST_DATASET_PATH
                           , delimiter=DELIMITER)

    # Apply pre-processing, already applied to train dataset
    test_dataset = process_dataset(dataset=test_dataset
                                   , compression_factor=COMPRESSION_FACTOR
                                   , normalize=NORMALIZE
                                   , derivative=DERIVATIVE)

    # Predict class indexes of a test dataset
    predicted_class_indexes = cdp2.predict(test_dataset)

    # Check how many of predicted class indexes is correct 
    matching_count = np.sum((np.array(predicted_class_indexes) == test_dataset.class_indexes))
    logger.info(f"Accuracy: {100*round(matching_count/len(predicted_class_indexes), 4)}%")


if __name__ == "__main__":
    train()
    test()

</pre>

### Performance - accuracy and training time  
CDP model has very small training time- it vary from seconds to minutes for dataset from USR database. 
Table below shows some elapsed training time and corresponding accuracy along with used hyper-parameters. 
Also, Fig. 1 shows comparison of the CDP method in terms of accuracy with some state-of-the-art time series 
classification method. 
Note: Accuracies reported for Fig.1 were obtained by **[C# implementation](https://github.com/naviveztim/CDP_C_Sharp)** of CDP method (for questions and inquiries: cdp_project@outlook.com). 
Table 1 contain training time and accuracies obtained by python implementation of the CDP method and Table 2 corresponding performance parameters
from C# implementation. 
Present Python implementation does not use any acceleration techniques such as numba, or multiprocessing. 

Table 1. Training time and accuracy of **python implementation** with numba of CDP method

| UCR Dataset | Num. classes | Num. train samples | Num. test samples | Training time, [sec] | Accuracy, [%] | Compression rate | Num. decision trees | Normalize | Derivative |
|-------------|--------------|--------------------|-------------------|----------------------|---------------|------------------|---------------------|-----------|------------|
| SwedishLeaf | 15           | 500                | 625               | 99                   | 85.4%         | 2                | 500                 | No        | No         |
| Beef        | 5            | 30                 | 30                | 43                   | 70.1%         | 1                | 200                 | Yes       | Yes        |
| OliveOil    | 4            | 30                 | 30                | 35                   | 76.6%         | 2                | 200                 | Yes       | No         |
| Symbols     | 6            | 25                 | 995               | 62                   | 86.9%         | 4                | 600                 | Yes       | Yes        |
| OsuLeaf     | 6            | 200                | 242               | 98                   | 90.1%         | 4                | 800                 | Yes       | Yes        |

There is also an implementation of CDP algorithm in C#, which on the same CPU produced even better results 
(Table 2)

Table 2. Training time and accuracy of **C# implementation** of CDP method

| UCR Dataset | Num. classes | Num. train samples | Num. test samples | Training time, [sec] | Accuracy, [%] | Compression rate | Num. decision trees | Normalize | Derivative |
|-------------|--------------|--------------------|-------------------|----------------------|---------------|------------------|---------------------|-----------|------------|
| SwedishLeaf | 15           | 500                | 625               | 16                   | 92.7%         | 2                | 700                 | No        | No         |
| Beef        | 5            | 30                 | 30                | 24                   | 86.8%         | 1                | 400                 | Yes       | Yes        |
| OliveOil    | 4            | 30                 | 30                | 71                   | 90.1%         | 2                | 200                 | Yes       | No         |
| Symbols     | 6            | 25                 | 995               | 4                    | 95.6%         | 4                | 600                 | Yes       | Yes        |
| OsuLeaf     | 6            | 200                | 242               | 15                   | 88.9%         | 4                | 800                 | Yes       | Yes        |

We tested several methods for time series classification on 40 datasets from UCR database. CDP methods stays well in terms 
of accuracy as shown on figure below. 
![Accuracy comparison](Accuracy_comparison.png)

Fig. 1 Comparison of state-of-the-art classifiers and CDP method. Used **C# implementation** of 
CDP method.    

### Model
Two files are produced during training process. First one contains representation in .pickle format
of decision tree sequence, and the second one (in csv format), contains concatenated decision patterns produced from decision
trees, for each time series from train dataset, as shown in the example below. 

<pre>
class_index,class_pattern
1,LLRLRLLRRLLLRLLLLRL...
1,LLLLRRRRLLLLLLRRRRR...
2,LLLLRRRRLLLLLLLLLLL...
</pre>

These files are stored in model folder given as an input parameter to the process. They have hardcoded names
(defined in cdp.py) as follows: 
<pre>
# Filename of trained model - contains sequence of decision trees
MODEL_FILENAME = 'cdp_model.pickle'
# Filename of csv file that contains predicted class indexes
PATTERNS_FILE_NAME = 'patterns.csv'
</pre>

### Classification
Currently, classification is done by producing decision pattern of an incoming time series, and comparing 
that pattern to such patterns from train dataset. The pattern from train dataset, which mostly resemble the 
incoming time series pattern will define its index. 

Default process of classification is a bit slow as the incoming time series pattern has to be compared 
with many patterns, which is a bit slow process. 

More advanced classification methods such as Neural Networks, Random Forests or other could be applied
for even more precise and fast classification, by taking produced decision patterns as input features
to these methods. 

### Contacts: 
cdp_project@outlook.com

### References: 

“Concatenated Decision Paths Classification for Datasets with Small Number of Class Labels”, Ivan Mitzev and N.H. Younan, ICPRAM, Porto, Portugal, 24-26 February 2017_

“Concatenated Decision Paths Classification for Time Series Shapelets”, Ivan Mitzev and N.H. Younan, International journal for Instrumentation and Control Systems (IJICS), Vol. 6, No. 1, January 2016_

“Combined Classifiers for Time Series Shapelets”, Ivan Mitzev and N.H. Younan, CS & IT-CSCP 2016 pp. 173–182, Zurich, Switzerland, January 2016_

“Time Series Shapelets: Training Time Improvement Based on Particle Swarm Optimization”, Ivan Mitzev and N.H. Younan, IJMLC 2015 Vol. 5(4): 283-287 ISSN: 2010-3700_


