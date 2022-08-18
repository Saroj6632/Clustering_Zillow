import pandas as pd
import env



''' function to connect to CodeUp SQL database'''
def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{db}'

def get_zillow_data():
    query= '''
            SELECT  
            bathroomcnt as bathroom,
            bedroomcnt as bedroom,
            basementsqft,
            parcelid,
            buildingqualitytypeid as quality_type, 
            calculatedfinishedsquarefeet as finished_square_ft, 
            garagecarcnt as garage,
            fips as county, 
            latitude, 
            longitude, 
            lotsizesquarefeet as lot_square_ft, 
            regionidcity as city, 
            poolcnt as pool,
            yearbuilt, 
            structuretaxvaluedollarcnt as structure_value, 
            taxvaluedollarcnt as house_value,
            landtaxvaluedollarcnt as land_value, 
            taxamount as tax, 
            logerror as log_error,
            transactiondate as transaction_date
            FROM properties_2017
            JOIN predictions_2017 USING (parcelid)
            WHERE transactiondate < '2018'
            AND propertylandusetypeid = 261
            AND longitude IS NOT NULL
            AND latitude IS NOT NULL
            
            '''
    
    
    df = pd.read_sql(query, get_connection('zillow'))

    return df