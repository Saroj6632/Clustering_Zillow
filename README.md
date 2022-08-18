# Zillow Clustering Project
## Goals
The project aims to improve original estimate of the log error by using clustering methodologies.

## Project Description
Zillow data for single family property with the transaction date in the year of 2017 will be analyzed and clustering methodologies will be used to predict the log error and various regression model will be built based upon the best features. And finally, I would give some recommendations, next steps to take.

## Deliverables
- Github repository
  1. README file
  2. Final notebook
  3. .py modules with all functions to reproduce the model

## Initial Questions
  1. Does county(location) is related to the logerror?
  2. Is there  any relation between log error and house price, yearbuilt?
  3. Is there any relation between log error  bead and bath ratio?
  4. Why Orange county have high log error?

## Data Dictionary
  -  bedroom : Number of bedrooms in home.
  -  bathroom : Number of bathrooms  
  -  parcelid : Unique identifier  
  -  county : Federal Information Processing Standard code. 
  -  yearbuilt: The Year house was built. 
  -  finished_square_ft : Calculated total finished living area of the home. 
  -  lot_square_ft : Area of the lot in square feet. 
  -  house_value : The total tax assessed value of the property. 
  -  structure_value : The assessed value of the built structure on the parcel. 
  -  land_value : The assessed value of the land area of the parcel 
  -  tax : The total property tax assessed for that assessment year. 
  -  garage : Total number of garages on the lot including an attached garage. 
  -  land_dollar_per_sqft : land taxvalue divided by lot size
  -  structure_dollar_per_sqft : structure taxvalue divided by square feet
  -  quality_type : Overall assessment of condition of the building from best (lowest) to worst (highest). 
  -  latitude : Latitude of the middle of the parcel multiplied by 10e6. 
  -  longitude : Longitude of the middle of the parcel multiplied by 10e6. 
  -  city : City in which the property is located (if any). 
  -  log_error : log error=log(Zestimate)−log(SalePrice). transaction_date : House transaction date.

# Steps to Reproduce My work
In order to reproduce my final report and the model following process should be followed

- env.py file that has credentials for successful connection with CodeUp DB Server.
- clone my project repo(including aquire.py and prepare.py).
- libraries to be used are pandas, matplotlib, seaborn, numpy, sklearn
- finally you should be able to run final_project report.


# Pipeline

## Acquire Data
  acquire.py module has all necessary functions to to connect to  SQL database and generate the zillow dataframe.

## Data Preparation
### prepare.py module has all the functions 
    1. address missing null values by dropping columns missing 60% and rows missing 75%
    2.Handle Outliers function to handle some extreme values in the columns. Some outliers in columns like bedroom , bathroom and house values were manualy     handled.
    3. Convert the latitude and longitude
    4. engineer new features like yearbuilt, structure_dollar_per_sqft, land_dollar_per_sqft 

## Data Split
  data was split into train, test and validate samples

## Data Imputation
  using impute function some columns with missing data were imputed

## Data Exploration
  Goal: Address the initial questions through visualizations and conducting statististics test.

## Clustering
  Three different cluster with different features created
  1. Cluster1 using longitude, latitude and county_dummies(Los_Angeles, Orange, Ventura)
  2. cluster2 using structure value, land value and house value
  3. cluster3 using yearbuilt, bed and bath ratio, structure_dollar_per_sqft, land_dollar_per_sqft
  

## Feature Selection
  Select k best and recursive feature elimantion method were used to select best features.

##  Modeling and Evaluation
  Regression models were developed to beat the baseline model.
  Models created: 
    1.linear Regression
    2. Lasso-lars
    3. Tweedieregressor
    4.2nd Degree Polynomial
    5. Interaction only polynomial
    
- RMSE scores calculated for each model
## Evaluate on best performing model
  test RMSE = 0.17
  
 ## Conclusion
  ## Summary
  none of my models developed were able to beat the baseline model. And also the clusters created were not that of useful for the project.
  
 ## Recommendations
  1. Different appraoch for creating clusters that could be helpful in reducing logerror.
  2. Explore more to create meaningful clusters
 
 ## Next Steps
  1. Change the parameters on models to see if any model perform best.
  2. may be linear regression model is the not the best approach here so I would like to build different models.



  
    
