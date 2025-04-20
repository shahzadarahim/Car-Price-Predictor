# CAR PRICE PREDICTION

# __1. Project Goal__
In this project, my goal is to predict whether a car owner will be able to sell the car and, if so, estimate its market value.

# __2. Data__
This dataset consists of 10 columns and 10,000 rows. Below is an overview of the data:
1. __Brand__: _The car's manufacturer name_
2. __Model__: _The specific type of the car model_
3. __Year__: _The manufacturing year of the car_
4. __Engine Size__: _The engine's capacity_
5. __Fuel Type__: _The type of fuel the car uses_
6. __Transmission__: _The car's gear system_
6. __Mileage__: _The total distance the car has traveled_
8. __Doors__: _The number of doors in the car_
9. __Previous Owners__: _The count of past owners_
10. __Price__: _The car's selling price_

# __3. Steps__
In this project, we are working with a Regression model. The steps we plan to follow are:

1. Setup the environment
2. Data Cleaning
    * 2.1. Split Test data
    * 2.2. Remove Extra Columns
    * 2.3. Columns Typecasting
    * 2.4. Handle Missing Values
    * 2.5. Handle Duplicate Rows
    * 2.6. Numerical Sanity Check
    * 2.7. Categorical Sanity Check
3. Exploratory Data Analysis
    * 3.1. Descriptive Analysis
        * 3.1.1. Numerical Data
        * 3.1.2. Categorical Data
    * 3.2. Correlation Analysis
        * 3.2.1. Numerical Data (Pearson)
        * 3.2.2. Categorical Data (ANOVA)
4. Data Preprocessing
    * 4.1. Handle Outlier
        * 4.1.1. Outlier Detection
        * 4.1.2. Dealing with Outlier
    * 4.2. Encoding Categorical Data
        * 4.2.1. Mean Target Encoding
        * 4.2.2. One-Hot Encoding
5. Feature and Target Declare and Data Splitting
    * 5.1. Declare Features and Target
    * 5.2. Data Splitting (Train, Val)
6. Model Building
    * 6.1. Try Algorithm [Linear Regression]:
    * 6.2. Try Algorithm [Decision Tree Regressor]
    * 6.3. Try Algorithm [Random Forest Regressor]
    * 6.4. Try Algorithm [Support Vector Regression]
    * 6.5. Try Algorithm [XGBoost Regressor]
    * 6.6. Try Algorithm [CatBoost Regression]
    * 6.7. Try Algorithm [AdaBoost Regression]
    * 6.8. Model Selection
    * 6.10. Hyperparameter Tunning the Best Algorithm and Build Model 
    * 6.11. Evaluate Best Model Performance on Val Set (cross validation on train and apply on val)
        * 6.11.1. Analysis Evaluation Metrics
        * 6.11.2. Analysis Plots
        * 6.11.3. Analysis Fitting (Over/Under)
    * 6.12. Apply Model on Test Set
7. Model Deployment
    * 7.1. Build App
    * 7.2. Deploy on Cloud