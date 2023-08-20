## Python implementation of Concatenation Decision Paths (CDP)- fast and accurate method for time series classification 

### Overview 
Python implementation of the CDP algorithm posses following advantages: 
- **very fast** to (re)train (training time vary from seconds to minutes for datasets from [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/))
- **very simple** to maintain (consists of 8 python files, spread in two folders)
- produces **compact (~KB) models**, in comparison with large standard models (~MB)  
- maintains **high accuracy** and is comparable or in some cases even more accurate than state of the arts algorithms (Fig.1) 
- python implementation does not depend on other machine learning package. It has only dependencies on standard python packages

![Accuracy comparison](Accuracy_comparison.png)

Fig. 1 Comparison of classifiers' accuracy  

### Installation 

### Training 

<pre>
# Obtain train dataset from UCR format file
train_dataset = from_ucr(filepath=<'train file path'>, delimiter=<'delimiter'>)

# Initialize CDP
cdp = CDP(dataset=train_dataset
          , model_folder=<'model folder path'>
          , num_classes_per_tree=2
          , pattern_length=100
          , compression_factor=1
          , derivative=False
          , normalize=False)

cdp.fit()
</pre>

### Testing 

<pre>
# Initialize CDP
cdp = CDP(dataset=None
          , model_folder=<'model folder path'>
          , num_classes_per_tree=2
          , pattern_length=100
          , compression_factor=1
          , derivative=False
          , normalize=False)

# Get already trained model 
cdp.load_model()

# Obtain test dataset from UCR format file 
test_dataset = from_ucr(<'test file path'>, delimiter=',')

# Predict class indexes of a test dataset
predicted_class_indexes = cdp.predict(test_dataset)

# Iterate through predicted indexes and check correspondence with the original
num_correct_predictions = 0
for i, row in test_dataset.iterrows():
    if row['class_index'] == predicted_class_indexes[i]:
        num_correct_predictions += 1

print(f"Accuracy: {100 * round(num_correct_predictions / len(predicted_class_indexes), 2)}%")

</pre>

### References: 

_“Concatenated Decision Paths Classification for Datasets with Small Number of Class Labels”, Ivan Mitzev and N.H. Younan, ICPRAM, Porto, Portugal, 24-26 February 2017_

_“Concatenated Decision Paths Classification for Time Series Shapelets”, Ivan Mitzev and N.H. Younan, International journal for Instrumentation and Control Systems (IJICS), Vol. 6, No. 1, January 2016_

_“Combined Classifiers for Time Series Shapelets”, Ivan Mitzev and N.H. Younan, CS & IT-CSCP 2016 pp. 173–182, Zurich, Switzerland, January 2016_

_“Time Series Shapelets: Training Time Improvement Based on Particle Swarm Optimization”, Ivan Mitzev and N.H. Younan, IJMLC 2015 Vol. 5(4): 283-287 ISSN: 2010-3700_


