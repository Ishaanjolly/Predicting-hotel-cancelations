# Libraries required
import numpy as np 
import pandas as pd
import pycountry as pc
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, plot_confusion_matrix


def find_na(df):
    
    '''
    This function will find the NA values in a dataframe and will return a table
    with the summary of the columns with this problem

    input:
    df: dataframe

    output
    df_na: summary dataframe for the columns with NA values
    '''
    
    # Find NA in the dataframe
    cols_na = [col for col in df.columns if df[col].isnull().any()]
    num_na = [i for i in df.isnull().sum() if i >0]

    # Prepare dataframe with a summary of the columns with NA
    df_na = pd.DataFrame(data = zip(cols_na, num_na), 
    columns = ["column_name", "na_values"]) 
    df_na["not_na"] = df.shape[0] - df_na["na_values"]
    df_na["proportion_na"] = df_na["na_values"] / df.shape[0]
    df_na["propotion_not_na"] = df_na["not_na"] / df.shape[0]

    return df_na

def impute_values(df):

    '''
    This function cleans the NA values present in a dataframe 

    If the column has numerical values, then it will fill the NA's with the median value.
    Else, it will fill the NA's with the mode value (categorical variables)

    input:
    df: pandas dataframe to clean

    output:
    df: cleaned pandas dataframe
    '''

    # Get the column names with NA values
    cols_na = [col for col in df.columns if df[col].isnull().any()]

    # Store the values to use for filling the NA's in a temporary dictionary
    values_dict = {}
    for i in cols_na:

        if df[i].dtype == "float64":
            clean_value = df[i].median()

        else:
            clean_value = df[i].mode()[0]
            
        values_dict[i] = clean_value
    
    # Replace the NA's with the dictionary
    df.fillna(value = values_dict, inplace = True)

    return df

def get_country_name(alpha_3):
    '''
    This function will get the country name from a given country code in alpha3 
    format using the Pycountry library

    input:
    alpha_3: list containing the country codes in alpha3 format

    output:
    country.name: Country name
    '''

    # Get the country names from the Pycountry library
    for country in list(pc.countries):
        if alpha_3 in country.alpha_3:
            return country.name
          
    return None

def add_country_names(df):
    '''
    This function will add the country names as a column into the booking 
    dataframe for a better understanding 

    input:
    df: bookings dataframe

    output:
    df: bookings dataframe with a new "country_name" column 
    '''

    # Get the country codes present in the dataframe
    country_codes = df["country"].unique().tolist() 

    # Get the country names as a list
    countries = [get_country_name(c) for c in country_codes]

    # Create a dataframe with the country names and join it to the bookings 
    # dataframe
    country_names_df = pd.DataFrame(
        {'country': country_codes, 'country_name' : countries})
    df = df.set_index('country').join(
        country_names_df.set_index('country')).reset_index('country')

    return df

def create_binaries(df, vars):
    '''
    This function will create binary variables for a given list of variables in
    a dataframe

    input
    df: dataframe
    vars: list of variables to create binary variables
    
    output
    df: dataframe with the binary variables
    
    '''
    for i in vars:

        df[f'binary_{i}'] = np.where(df[i]>0, 1,0)
    
    return df

def count_how_many(df, feature):
    '''
    This function find how many observations of this feature was cancelled 

    input
    feature: a column from the dataframe

    output:
    Feature: dataframe with the number of cancellations
    '''
    Feature = df[['is_canceled',feature]].groupby(['is_canceled',feature]).size().reset_index(name = 'count')
    
    return Feature

def count_how_many2(feature):
    '''
    This function find how many observations of this feature was cancelled 

    input
    feature: a column from the dataframe

    output:
    Feature: dataframe with the number of cancellations
    '''
    Feature = train_set[['is_canceled',feature]].groupby(['is_canceled',feature]).size().reset_index(name = 'count')
    
    return Feature

def roc_plot(y_true, y_pred):
    """ Draw an ROC curve and report AUC
    """
    roc = pd.DataFrame(
    data = np.c_[sklearn.metrics.roc_curve(y_true, y_pred)],
    columns = ('fpr', 'tpr', 'threshold')
    )
    sns.lineplot(x='fpr', y='tpr', data=roc, ci=None)
    plt.plot([0,1],[0,1], 'k--', alpha=0.5) # 0-1 line
    plt.title("ROC curve (auc = %.4f)" % sklearn.metrics.roc_auc_score(y_true, y_pred))
    plt.show()

def model_assessment(model, X, y):

    cm = plot_confusion_matrix(model, X, y, display_labels = ["not_canceled", "canceled"], cmap = plt.cm.Blues, normalize = "true")
    y_pred = model.predict(X)
    print("Accuracy score: ", accuracy_score(y, y_pred))

    return cm