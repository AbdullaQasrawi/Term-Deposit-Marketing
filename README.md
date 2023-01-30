# Term-Deposit-Marketing    
This project is focused on predicting the **customers who are more likely to buy an investment product** by applying machine learning algorithms on the given data. The data contains information about the customers and their investment behavior. The problem is framed as a **binary classification problem** where the target variable 'y' is either 'yes' or 'no'.    
# Data Preprocessing  
The first step in the project is to preprocess the data. The data is read into a pandas dataframe and any 'unknown' values in the data are replaced with 'nan' values. The data are highly skewed so i try to keep the values of class 'yes' which is limited  then i all nans thet corespond with class nor removed while nans that corespond with class yes are filled with most frequent value within its class.  
# Encoding
The categorical columns in the data are then encoded using label encoder. The label encoder method is defined in the 'enc' function. The categorical columns are transformed into numerical values and the transformed values are stored in the dataframe. but then dummy variables are used and the result enhanced well.  
# Outlier Detection
Outliers are detected using Z-scores and the rows with Z-scores greater than 3 for class 0 are dropped from the data. and the class is keeping due to the limited amount of data for this class.  
# Split into Training and Testing data  
The data is then split into training and testing data using the train_test_split method. The data is split in the ratio of 85:15 for training and testing data respectively with stratified sampling.

# Feature Scaling  
The data is then standardized using the StandardScaler method from the scikit-learn library. The integer columns 'age', 'balance', 'duration', and 'day' are transformed using the StandardScaler method.

# Model Building
The machine learning algorithm used in this project is Random Forest Classifier. The hyperparameters of the model are tuned using the GridSearchCV method. The scoring metric used in this project is the F1 score.

# Evaluation
The performance of the model is evaluated using the classification report and confusion matrix. The classification report provides the precision, recall, and F1 score of the model. The confusion matrix provides a visual representation of the true positive, false positive, true negative, and false negative predictions made by the model.
