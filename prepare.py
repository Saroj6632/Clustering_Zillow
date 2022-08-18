import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing




def nulls_by_col(df):
    ''' This function returns the number of data misssing in each column and the percenatage missing'''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)



def nulls_by_row(df):
    '''function to count of number of missing values in rows.'''
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['parcelid', 'num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)


#
def handle_missing_values(df, prop_required_columns=0.6, prop_required_row=0.75):
    ''' handle missing values by dropping the columns and rows using proportion threshold'''
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df



# handle outliers function to reduce the noise

def iqr_outliers(df, k=1.5, col_list=None):
    if col_list != None:
        for col in col_list:
            q1, q3 = df[col].quantile([.25, .75])
            iqr = q3 - q1
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    else:
        for col in list(df):
            q1, q3 = df[col].quantile([.25, .75])
            iqr = q3 - q1
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df


def clean_zillow(df):
    df = df.sort_values('transaction_date').drop_duplicates('parcelid',keep='last')
    # remove duplicates for the tranaction date of the property
    df = df[(df['bedroom'] > 0) & (df['bathroom'] > 0)]
    # properties with bed and bath greater than 0
    
    #handle missing values
    df = handle_missing_values(df)
    df.county = df.county.astype(int)
    df['county'] = np.where(df.county == 6037, 'Los_Angeles',
                           np.where(df.county == 6059, 'Orange', 
                                   'Ventura'))    
    # columns with outliers needed to be addressed
    outlier_cols = ['finished_square_ft', 'lot_square_ft', 'structure_value', 'house_value', 'land_value','tax']
    
    #handle outliers
    df = iqr_outliers(df, col_list=outlier_cols)
    df = df[df.bedroom <= 6]
    df = df[df.bathroom <= 6]
    df = df[df.house_value < 2500000]
    df.latitude = df.latitude / 1_000_000
    df.longitude = df.longitude / 1_000_000
    #recalculate yearbuilt to age of home:
    df.yearbuilt = 2017 - df.yearbuilt 
    df['age_bin'] = pd.cut(df.yearbuilt, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = ["0-5","5-10","10-20","20-30", "30-40", "40-50", "50-60", "60-70", "70-80", 
                                     "80-90", "90-100", "100-110", "110-120", "120-130", "130-140"])

    # create taxrate column
    df['taxrate'] = df.tax/df.house_value*100
    # create  bed bath ratio column
    df['bath_bed_ratio'] = df.bathroom/df.bedroom
    #bin land tax value
    df['land_tax_value_bin'] = pd.cut(df.land_value, bins = [0, 50000, 100000, 150000, 200000, 250000,350000, 450000, 650000, 800000, 1000000], labels = ["< $50,000","$100,000", "$150,000", "$200,000", "$250,000", "$350,000", '$450,000', "$650,000", "$800,000", "$1,000,000"])
     # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structure_value/df.finished_square_ft
    # bin structure dollar per sq ft
    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )
    
    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.land_value/df.lot_square_ft

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = ['0', '1', '5-19', '20-49', '50-99', '100-249', '250-499', '500-999', '1000-1499', '1500-2000']
                                      )
    return df



# function to split data

def split_data(df):
    '''this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.'''
    
    
    train_validate, test = train_test_split(df,test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate,test_size=0.3, random_state=123)

    return train, validate, test
    
def xy_split(train, validate,test):
    # split train into X_train and y train
    x_train = train[['finished_square_ft','latitude','longitude','lot_square_ft','bedroom','bathroom','structure_dollar_per_sqft',
                 'structure_value','house_value','land_value','yearbuilt','bath_bed_ratio','land_dollar_per_sqft']]
    y_train = train[['log_error']]

    # split validate into X and y validate
    x_validate = validate[['finished_square_ft','latitude','longitude','lot_square_ft','bedroom','bathroom','structure_dollar_per_sqft',
                 'structure_value','house_value','land_value','yearbuilt','bath_bed_ratio','land_dollar_per_sqft']]
    y_validate = validate[['log_error']]

    #split test into X and y test
    x_test= test[['finished_square_ft','latitude','longitude','lot_square_ft','bedroom','bathroom','structure_dollar_per_sqft',
                 'structure_value','house_value','land_value','yearbuilt','bath_bed_ratio','land_dollar_per_sqft']]
    y_test= test [['log_error']]




    return x_train,y_train, x_validate,y_validate, x_test, y_test



def impute_missing_value_zillow(df):
    ''' function to impute the missing values '''
    df['quality_type'] = df.quality_type.fillna(round(df.quality_type.mean()))
    df.dropna(inplace=True)
    return df



def scaled_data(x_train, x_validate, x_test, y_train, y_validate, y_test):

    # Make the scaler
    scaler = MinMaxScaler()

    # Fit the scaler
    scaler.fit(x_train)

    # Use the scaler
    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns,index=x_train.index)
    x_validate_scaled = pd.DataFrame(scaler.transform(x_validate), columns=x_validate.columns, index=x_validate.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    
    # Make y_values separate dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    #Unscaled data for later
    x_unscaled= pd.DataFrame(scaler.inverse_transform(x_test), columns=x_test.columns, index=x_test.index)
    return x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_validate, y_test, x_unscaled



def rfe1(X, y, k):
    '''
    Takes in the predictor (X_train_scaled), the target (y_train), 
    and the number of features to select (k).
    Returns the top k features based on the RFE class.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select =k)
    rfe.fit(X, y)
    
    features_to_select =  X.columns[rfe.support_].tolist()
    return features_to_select


def select_kbest(X,y,k):
    ''' Takes in the predictors (X_train_scaled), the target (y_train), 
    and the number of features to select (k) 
    and returns the names of the top k selected features based on the SelectKBest class'''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()].tolist()
