import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

def clean_zillow(df):
    '''
    Takes in a df of zillow_data and cleans the data appropriatly by dropping nulls,
    removing white space,
    creates dummy variables for Contract type,
    converts data to numerical, and bool data types, 
    and drops columsn that are not needed.
    
    return: df, a cleaned pandas data frame.
    '''
    
    # Instead of using dummies to seperate contracts use, 
    # df[['Contract']].value_counts()
    # Use a SQL querry
    
    df = df
    df = df.loc[:, df.isnull().mean() < .80]
    df = df.fillna(0)
    df = df.dropna()
    
    return df
    
def split_zillow(df):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test

def split(df, stratify_by=''):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['taxvaluedollarcnt'])
    y_train = train['taxvaluedollarcnt']
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['taxvaluedollarcnt'])
    y_validate = validate['taxvaluedollarcnt']
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['taxvaluedollarcnt'])
    y_test = test['taxvaluedollarcnt']
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


# taxvaluedollarcnt/taxamount