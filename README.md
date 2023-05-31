# Funding_Deep_Learning

- - -

## Overview of the Analysis

- Purpose
    - The purpose of this neural network model is to determine, based off the feature columns, if funding granted was successfully used by the recipients. 

- What is being predicted?
    - The `IS_SUCCESSFUL` column is the target column and will be what the model will be attempting to predict.

- Goal
    - The goal is to break down the data in preprocessing to keep as much explained variance as possible and to then create the most effective and efficent Tensorflow neural network as possible to predict the target column.

- Measuring
    - The model will use `TrainTestSplit` to get training data and testing data from the original dataset.
    - Training and testing data will be scaled using `StandardScaler`.
    - The preprocessing pipeline will include `Lasso` and `Ridge` tests to drop low corrilated columns.
    - Training and testing data will go through a `PCA` to reduce columns even further.
    - The model training will be evaluated based off accuracy and loss.
    - The model testing will be valided against the testing data as `val_accuracy`.

## Results

- Data Preprocessing

    - What variable(s) are the target(s) for your model?
        - The target variable is the `IS_SUCCESSFUL` column in the dataset which has either a `1` or a `0` meaning successful use of funding or not respectively.

    - What variable(s) are the features for your model?
        - After dropping the `IS_SUCCESSFUL`, `EIN`, and `NAME` columns, the remaining columns serve as variables/features in the subsequent models.

    - What variable(s) should be removed from the input data because they are neither targets nor features?
        - Specific variables prior to putting the data through a hot-ended filter are not removed from the dataset outside of `EIN`, and `NAME` columns meantioned earlier. Lasso and Ridge tests were compared to a PCA text to reduce the data size while retaining as much of the explained variance as possible. Ultimately these two methods do similar preprocessing to the data so both were used together in conjunction with setting boundries using ~1.5x the interquartile range of the values in the `ASK_AMT` column to preprocess the data.

- Compiling, Training, and Evaluating the Model

    - How many neurons, layers, and activation functions did you select for your neural network model, and why?
        - While preprocessing, I maintained using the default prescribed model parameters of 80 neurons on the initial layer, 30 for the 1 hidden layer, and 1 for the output layer using `relu`, `relu`, `sigmoid` respectively. 
        - To find the optimal selection of neurons, layers, and activation functions, I used Keras Tuner in two phases: 
        - 1. limiting down and eliminating activation functions that did not improve the model.
            - The tuner was allowed to select any activation function for the first and hidden layers. The output layer was always 1 neuron with a `sigmoid` activation function. `leaky_relu`, `tanh`, `selu`, `elu`, & `sigmoid` were all eliminated from the pool based off the tuner trials. `relu` was eliminated from the initial layer but not from the hidden layers.
        - 2. test and find the optimal layout of neurons and hidden layers. 
            - Finally, the tuner was allowed to trial test to find the optimal layout from 3-8 hidden layers and 30-80 neurons per layer. Any further hidden layers did not result in any benefit and any further neurons contributed to a model that required too much computing power. The chosen layout for the model was the one with the highest `val_accuaracy` and can be found in the ipynb or just below:

            ``` 
            initial activation: tanh
            first_units: 80
            num_layers: 3
            layer_activation_0: tanh
            units_0: 50
            layer_activation_1: relu
            units_1: 55
            layer_activation_2: relu
            units_2: 75
            layer_activation_3: tanh
            units_3: 30
            ``` 

    - Were you able to achieve the target model performance?
        - The target model performance was hit in the preprocessing. Any neuron, layer, or activation function modification caused minimal impact past 75% `val_accuracy`.

    - What steps did you take in your attempts to increase model performance?
        - methods used include:
        1. Lasso and Ridge tests to drop low impact columns.
        2. PCA the data to reduce scaled data columns down to 35 columns for 100% explained variance, 20 columns for 70% explained variance with a model performance boost.
        3. Oversampling by comparing the use of RandomOverSampler and SMOTE oversampler. SMOTE in this case caused more inaccuracy in the validation test. It is possible that the synthetic samples are not accurately respresenting the actual data. RandomOverSampler copying samples did increase the accuracy marginally.
        4. Altering the model's activation function, layer count, and per layer neuron count via Keras Tuner.

## Summary

* The neural network model did not see any improvement as it became more complex. The target accuracy of 75% was reached but I believe a different model like RandomForest may perform better as the target column of success is a result of a string of decisions that the correlation captured by the tensorflow neural network is not able to identify. In terms for a decision maker, a neural network has the con of having processing being hidden and with each neuron having different weights, it is very difficult to show the corrilation between two columns or a feature column to the target column.

- - -

### Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

    - `EIN and NAME`—Identification columns
    - `APPLICATION_TYPE`—Alphabet Soup application type
    - `AFFILIATION`—Affiliated sector of industry
    - `CLASSIFICATION`—Government organization classification
    - `USE_CASE`—Use case for funding
    - `ORGANIZATION`—Organization type
    - `STATUS`—Active status
    - `INCOME_AMT`—Income classification
    - `SPECIAL_CONSIDERATIONS`—Special considerations for application
    - `ASK_AMT`—Funding amount requested
    - `IS_SUCCESSFUL`—Was the money used effectively

- - -

### Instructions

#### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the feature(s) for your model?

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

7. Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

#### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

#### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

- Use any or all of the following methods to optimize your model:

    - Adjust the input data to ensure that no variables or outliers are - causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
    - Add more neurons to a hidden layer.
    - Add more hidden layers.
    - Use different activation functions for the hidden layers.
    - Add or reduce the number of epochs to the training regimen.

1. Create a new Google Colab file and name it `AlphabetSoupCharity_Optimization.ipynb`.
// I ran optimization in the same ipynb file as the initial model.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

#### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. Overview of the analysis: Explain the purpose of this analysis.

2. Results: Using bulleted lists and images to support your answers, address the following questions:

- Data Preprocessing

    - What variable(s) are the target(s) for your model?
    - What variable(s) are the features for your model?
    - What variable(s) should be removed from the input data because they are neither targets nor features?

- Compiling, Training, and Evaluating the Model

    - How many neurons, layers, and activation functions did you select for your neural network model, and why?
    - Were you able to achieve the target model performance?
    - What steps did you take in your attempts to increase model performance?

3. Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

#### Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.

1. Download your Colab notebooks to your computer.

2. Move them into your Deep Learning Challenge directory in your local repository.

3. Push the added files to GitHub.

- - - 

## References

IRS. Tax Exempt Organization Search Bulk Data Downloads. [https://www.irs.gov/](https://www.irs.gov/).

- - - 

© 2023 edX Boot Camps LLC