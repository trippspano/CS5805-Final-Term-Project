import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def clean_dataset():
    # Load the dataset
    file_path = "data/athlete_events.csv"  # Replace with your dataset path
    df = pd.read_csv(file_path)

    # Check for missing values
    missing_values = df.isnull().sum()
    print(missing_values)

    # What columns have missing values?
    # for column in df.columns:
    #     if df[column].isnull().sum() > 0:
    #         print(f"\nFirst 5 missing values in column '{column}':")
    #         print(df[df[column].isnull()][column].head(5))


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

def standardize(df):
    scaler = StandardScaler()
    # Exclude specific columns from standardization
    columns_to_exclude = ['is_male', 'is_summer']
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in columns_to_exclude]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def balance_medals(df):
    # Separate the dataset into different classes
    gold = df[df['Medal_encoded'] == 3]
    silver = df[df['Medal_encoded'] == 2]
    bronze = df[df['Medal_encoded'] == 1]
    no_medal = df[df['Medal_encoded'] == 0]

    # Find the size of the largest class
    max_size = max(len(gold), len(silver), len(bronze), len(no_medal))

    # Resample each class to match the size of the largest class
    gold_resampled = resample(gold, replace=True, n_samples=max_size)
    silver_resampled = resample(silver, replace=True, n_samples=max_size)
    bronze_resampled = resample(bronze, replace=True, n_samples=max_size)
    no_medal_resampled = resample(no_medal, replace=True, n_samples=max_size)

    # Combine the resampled classes back into a single dataset
    balanced_df = pd.concat([gold_resampled, silver_resampled, bronze_resampled, no_medal_resampled])

    return balanced_df


# df = encoding("Weight")
# df, scaler = standardize(df)
# print(df.head().to_string())
