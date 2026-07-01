import pandas as pd

def clean_data(df):

    columns = [
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI'
    ]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(0, df[col].mean())

    df.drop_duplicates(inplace=True)

    return df
