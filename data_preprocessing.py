# This file handles cleaning the dataset and encoding

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Cleans the dataset
# Finds out which columns have missing values and fills them appropriately
def clean_dataset():

    file_path = "data/athlete_events.csv"  # Replace with your dataset path
    df = pd.read_csv(file_path)

    # Check for missing values
    # missing_values = df.isnull().sum()
    # print(missing_values)



    # Fill missing values with the mean of the column for Age, Height, and Weight
    df.fillna({"Age":df["Age"].mean()}, inplace=True)
    df.fillna({"Height":df["Height"].mean()}, inplace=True)
    df.fillna({"Weight":df["Weight"].mean()}, inplace=True)

    # Fill NA values in Medal column with "No Medal"
    df.fillna({"Medal":"No Medal"}, inplace=True)

    # Show that there are no more missing values
    missing_values = df.isnull().sum()
    print(missing_values)

    # Save the cleaned dataset
    df.to_csv("data/athlete_events_cleaned.csv", index=False)

# Encoding and standardizing the data
# Target is used for target encoding
def encoding(target: str):
    df = pd.read_csv("data/athlete_events_cleaned.csv")
    # print(df.head())

    # Drop ID
    df.drop(columns=['ID'], inplace=True)

    # Drop Name
    df.drop(columns=['Name'], inplace=True)

    # Boolean encoding for Sex
    df['is_male'] = df['Sex'].map({'M': 1, 'F': 0})
    df.drop(columns=['Sex'], inplace=True)

    # Leave age, height, and weight

    # Target encoding for Team
    team_means = df.groupby('Team')[target].mean()
    df['Team_encoded'] = df['Team'].map(team_means)
    df.drop(columns=['Team'], inplace=True)

    # Drop NOC
    df.drop(columns=['NOC'], inplace=True)

    # Drop Games
    df.drop(columns=['Games'], inplace=True)

    # Leave Years

    # Boolean encoding for Season
    df['is_summer'] = df['Season'].map({'Summer': 1, 'Winter': 0})
    df.drop(columns=['Season'], inplace=True)

    # Drop city
    df.drop(columns=['City'], inplace=True)

    # Target encoding for Sport
    sport_means = df.groupby('Sport')[target].mean()
    df['Sport_encoded'] = df['Sport'].map(sport_means)
    df.drop(columns=['Sport'], inplace=True)
    # unique_sports_count = df['Sport'].nunique()
    # print(f"Number of unique sports: {unique_sports_count}")

    # Target encoding for Event
    event_means = df.groupby('Event')[target].mean()
    df['Event_encoded'] = df['Event'].map(event_means)
    df.drop(columns=['Event'], inplace=True)
    # unique_events_count = df['Event'].nunique()
    # print(f"Number of unique events: {unique_events_count}")

    # Label encoding for Medal
    medal_mapping = {
        'Gold': 3,
        'Silver': 2,
        'Bronze': 1,
        'No Medal': 0
    }
    df['Medal_encoded'] = df['Medal'].map(medal_mapping)
    df.drop(columns=['Medal'], inplace=True)
    # print(df['Medal_encoded'].value_counts())

    return df

# Encoding for clustering, change is frequency encoding for Team, Sport, and Event
def encoding_no_target():
    df = pd.read_csv("data/athlete_events_cleaned.csv")

    # Drop ID
    df.drop(columns=['ID'], inplace=True)

    # Drop Name
    df.drop(columns=['Name'], inplace=True)

    # Boolean encoding for Sex
    df['is_male'] = df['Sex'].map({'M': 1, 'F': 0})
    df.drop(columns=['Sex'], inplace=True)

    # Leave age, height, and weight

    # Frequency encoding for Team
    team_freq = df['Team'].value_counts()
    df['Team_encoded'] = df['Team'].map(team_freq)
    df.drop(columns=['Team'], inplace=True)

    # Drop NOC
    df.drop(columns=['NOC'], inplace=True)

    # Drop Games
    df.drop(columns=['Games'], inplace=True)

    # Leave Years

    # Boolean encoding for Season
    df['is_summer'] = df['Season'].map({'Summer': 1, 'Winter': 0})
    df.drop(columns=['Season'], inplace=True)

    # Drop city
    df.drop(columns=['City'], inplace=True)

    # Frequency encoding for Sport
    sport_freq = df['Sport'].value_counts()
    df['Sport_encoded'] = df['Sport'].map(sport_freq)
    df.drop(columns=['Sport'], inplace=True)

    # Frequency encoding for Event
    event_freq = df['Event'].value_counts()
    df['Event_encoded'] = df['Event'].map(event_freq)
    df.drop(columns=['Event'], inplace=True)

    # Label encoding for Medal
    medal_mapping = {
        'Gold': 3,
        'Silver': 2,
        'Bronze': 1,
        'No Medal': 0
    }
    df['Medal_encoded'] = df['Medal'].map(medal_mapping)
    df.drop(columns=['Medal'], inplace=True)

    return df

# Encoding for association rules, bins are created for Age, Height, Weight, and Year
def encoding_association():
    df = pd.read_csv("data/athlete_events_cleaned.csv")

    # Drop ID
    df.drop(columns=['ID'], inplace=True)

    # Drop Name
    df.drop(columns=['Name'], inplace=True)

    # Drop NOC
    df.drop(columns=['NOC'], inplace=True)

    # Drop Games
    df.drop(columns=['Games'], inplace=True)

    # Drop city
    df.drop(columns=['City'], inplace=True)

    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 100], labels=['Under 20', '20-30', '30-40', '40+'])

    df['Height_Group'] = pd.cut(df['Height'], bins=5,
                                        labels=['Very Short', 'Short', 'Average', 'Tall', 'Very Tall'])
    df['Weight_Group'] = pd.cut(df['Weight'], bins=5,
                                        labels=['Very Light', 'Light', 'Medium', 'Heavy', 'Very Heavy'])

    # Create bins and labels for decades
    bins = list(range(1890, 2030, 10))  # Create bins for each decade
    labels = [f"{decade}s" for decade in range(1890, 2020, 10)]  # Create labels

    # Assign each year to its respective bin
    df['Decade'] = pd.cut(df['Year'], bins=bins, labels=labels, right=False)

    return df

# Standardize the data
def standardize(df):
    scaler = StandardScaler()
    # Exclude specific columns from standardization
    columns_to_exclude = ['is_male', 'is_summer']
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in columns_to_exclude]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
