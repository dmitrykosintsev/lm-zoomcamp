# This file can be used to train the model of choice from the Notebook.ipnyb.
# Chosen model: Random forest:
# max_depth = 5
# min_samples_leaf = 50
# n_estimators=100
# random_state=1

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pickle
from typing import cast, BinaryIO

# Function to process the Missing Not At Random values
def preprocess_cyberbullying_data(df):
    """
    Comprehensive preprocessing for cyberbullying dataset
    Returns two dataframes: df (all students, no missing values) and victims_df (only victims)
    """
    df = df.copy()

    # === 1. Handle conditional missingness - Fill original columns ===

    # Cyberbullying types: if not cyberbullied, mark as "Not Applicable"
    df['cyberbullying_types'] = df.apply(
        lambda row: 'Not Applicable' if not row['experienced_cyberbullying']
        else (row['cyberbullying_types'] if pd.notna(row['cyberbullying_types'])
              else 'None'),
        axis=1
    )

    # Reported incident: if not cyberbullied, mark as "Not Applicable"
    df['reported_incident'] = df.apply(
        lambda row: 'Not Applicable' if not row['experienced_cyberbullying']
        else ('Yes' if row['reported_incident'] == True
              else 'No' if row['reported_incident'] == False
        else 'Unknown'),
        axis=1
    )

    # Reported to: if not cyberbullied or didn't report, mark accordingly
    df['reported_to'] = df.apply(
        lambda row: 'Not Applicable' if not row['experienced_cyberbullying']
        else ('Not Reported' if row['reported_incident'] in ['No', False]
              else (row['reported_to'] if pd.notna(row['reported_to'])
                    else 'Unknown')),
        axis=1
    )

    # Mental health impact: if not cyberbullied, mark as "Not Applicable"
    df['mental_health_impact'] = df.apply(
        lambda row: 'Not Applicable' if not row['experienced_cyberbullying']
        else (row['mental_health_impact'] if pd.notna(row['mental_health_impact'])
              else 'None'),
        axis=1
    )

    # Impact certainty: already handled in generation, but clean up any issues
    df['impact_certainty'] = df.apply(
        lambda row: 'Not Applicable' if not row['experienced_cyberbullying']
        else (row['impact_certainty'] if pd.notna(row['impact_certainty'])
              else 'Unsure'),
        axis=1
    )

    # === 2. Create additional categorical variable for reporting status ===
    df['reported_status'] = df.apply(
        lambda row: 'Not Applicable' if not row['experienced_cyberbullying']
        else ('Reported' if row['reported_incident'] == 'Yes'
              else 'Not Reported'),
        axis=1
    )

    # === 3. Extract useful numeric counts from multi-value fields ===
    # Cyberbullying types count
    df['num_bullying_types'] = df['cyberbullying_types'].apply(
        lambda x: 0 if x in ['None', 'Not Applicable'] else len(str(x).split('; '))
    )

    # Create binary flags for each type (keep as boolean for readability)
    for bully_type in ['Insults', 'Defamation', 'Exclusion', 'Threats', 'Sexual Harassment']:
        col_name = f'experienced_{bully_type.lower().replace(" ", "_")}'
        # False for everyone initially, True only if cyberbullied AND experienced this type
        df[col_name] = (
                df['experienced_cyberbullying'] &
                df['cyberbullying_types'].str.contains(bully_type, na=False)
        )

    # Mental health impacts count
    df['num_mental_health_symptoms'] = df['mental_health_impact'].apply(
        lambda x: 0 if x in ['None', 'Not Applicable'] else len(str(x).split('; '))
    )

    # Create binary flags for each symptom (keep as boolean)
    for symptom in ['Insomnia', 'Academic Decline', 'Depression', 'Anxiety',
                    'Loss of Appetite', 'Suicidal Thoughts']:
        col_name = f'has_{symptom.lower().replace(" ", "_")}'
        df[col_name] = (
                df['experienced_cyberbullying'] &
                df['mental_health_impact'].str.contains(symptom, na=False)
        )

    return df

def save_model(dv, rf):
    # Saving the model
    output_file = f'model_d=5_msl=50.bin'

    with open(output_file, 'wb') as f_out:
        # noinspection PyTypeChecker
        # f_out = cast(BinaryIO, f_out)
        pickle.dump((dv, rf), f_out)

    print("Successfully saved model to disk")

df = pd.read_csv('./data/students_data.csv')

# Creating two separate datasets - all data and only data from victims of cyber bullying
df = preprocess_cyberbullying_data(df)

# Defining variables that will be used for prediction
risk_prediction = ['age_group', 'gender', 'daily_internet_hours', 'primary_activity',
    'uses_facebook', 'num_social_media_accounts', 'exposed_to_bad_language',
    'learned_bad_words', 'received_school_education', 'awareness_level']

# Splitting the dataset
df_risk_full_train, df_risk_test = train_test_split(df, test_size=0.2, random_state=1)
df_risk_train, df_risk_val = train_test_split(df_risk_full_train, test_size=0.25, random_state=1)
len(df_risk_train), len(df_risk_val), len(df_risk_test)

# Reseting the index of datasets
df_risk_val = df_risk_val.reset_index(drop=True)
df_risk_train = df_risk_train.reset_index(drop=True)
df_risk_test = df_risk_test.reset_index(drop=True)

# Defining what values we need to predict
y_train = df_risk_train.experienced_cyberbullying.values
y_val = df_risk_val.experienced_cyberbullying.values
y_test = df_risk_test.experienced_cyberbullying.values

# Make sure that the target value y is not in your dataframe.
# Deleting the value to predict to avoid using it in the model
del df_risk_train['experienced_cyberbullying']
del df_risk_val['experienced_cyberbullying']
del df_risk_test['experienced_cyberbullying']

# Train dataset
dv = DictVectorizer(sparse=False)
train_dicts = df_risk_train[risk_prediction].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

# Validation dataset
val_dicts = df_risk_val[risk_prediction].to_dict(orient='records')
X_val = dv.transform(val_dicts)

rf = RandomForestClassifier(n_estimators=10, random_state=1) # Fixing random state for the result to be reproducable
rf.fit(X_train, y_train)

save_model(dv, rf)