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
