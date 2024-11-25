# DSV-assignment
This repository contains my DSA Assignment tasks.

Tasks:

Dataset Exploration:
Load the Iris dataset from sklearn.datasets. Display the first five rows, the dataset’s shape, and summary statistics (mean, standard deviation, min/max values) for each feature. Explanation:

Load the dataset: We use load_iris() to get the Iris dataset.

Convert to DataFrame: This allows us to manipulate and view the dataset easily.

Display First Five Rows: iris_df.head() shows the first five rows.

Dataset Shape: iris_df.shape gives the number of rows and columns.

Summary Statistics:

iris_df.describe() shows mean, standard deviation, min, max, and quartiles for each feature.

Data Splitting:
Split the Iris dataset into training and testing sets using an 80-20 split. Print the number of samples in both the training and testing sets. Explanation

In this task, we split the Iris dataset into training and testing sets, which allows us to evaluate our model's performance on unseen data.

An 80-20 split means that 80% of the dataset will be used for training the model, and 20% will be used for testing.

Steps:

Load the Dataset: We start by loading the Iris dataset, which contains 150 samples. Each sample includes features about the iris flowers and a target label representing the species of the flower.

Split the Data: Using the train_test_split function from Scikit-Learn, we split the data into training and testing sets with an 80-20 ratio. This results in:

Training Set: 120 samples (80% of 150)

Testing Set: 30 samples (20% of 150)

Random State: We set random_state=42 to ensure the split is reproducible. This helps us get the same split every time we run the code.

The split provides:

Training Set (120 samples): Used to train the model, allowing it to learn the relationships within the data.

Testing Set (30 samples): Used to evaluate the model's performance on new, unseen data, helping us assess its accuracy and generalization ability.

Linear Regression:
Use a dataset with the features YearsExperience and Salary. Fit a linear regression model to predict Salary based on YearsExperience. Evaluate the model's performance using Mean Squared Error (MSE) on the test set. Explanation:

In this task, we'll work with a simple dataset that has two features: YearsExperience (the number of years someone has worked) and Salary (the corresponding salary for that experience). The goal is to use Linear Regression to predict the salary based on years of experience and evaluate the model's performance using Mean Squared Error (MSE).

Steps to accomplish this: Dataset: The dataset contains two variables:

YearsExperience: The number of years a person has been working. Salary: The salary corresponding to that experience. Example dataset:

YearsExperience Salary 1 40000 2 45000 3 50000 ... ... Model: We use Linear Regression to create a model. Linear regression assumes a linear relationship between the independent variable (YearsExperience) and the dependent variable (Salary). The model aims to find the line of best fit, which minimizes the difference between predicted and actual salaries.

Data Splitting: The dataset is split into a training set (for fitting the model) and a test set (for evaluating its performance). We commonly use an 80-20 or 70-30 split.

Fitting the Model: The linear regression model is trained on the training set, which means it learns the relationship between the number of years worked and the salary.

Prediction: The trained model is then used to predict salaries on the test set based on years of experience.

Evaluation (Mean Squared Error - MSE): After prediction, we calculate the Mean Squared Error (MSE), which is a metric to evaluate the model's performance. MSE measures the average squared difference between the actual values and the predicted values.# DSV-ASSIGNMENT
