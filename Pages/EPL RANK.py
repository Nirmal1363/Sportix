import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np


# List of encodings to try
encodings = ['utf-8', 'latin-1', 'cp1252']

df = None  # Initialize the DataFrame

# Try loading the dataset with different encodings
for encoding in encodings:
    try:
        df = pd.read_csv('EPLDataml.csv', encoding=encoding).dropna()
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise ValueError("Failed to load the dataset. Please check the file and its encoding.")

# Create a set to store all unique team names
squads = set(df['Squad'].astype(str))

# Sort the team names alphabetically
sorted_squads = sorted(list(squads))

# Create a dictionary to map teams to their respective integer values
squad_integer_map = {squad: index + 1 for index, squad in enumerate(sorted_squads)}

# Create a new column in the DataFrame to store the integer class values
df['Squad_no'] = df['Squad'].map(squad_integer_map)

# Select relevant independent variables (features) for prediction
features = ['MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Pts/MP', 'Attendance']

# Select the target variable (ranking)
df['Rk'] = pd.to_numeric(df['Rk'], errors='coerce', downcast='integer')
target = 'Rk'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Initialize and train the Random Forest regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Sort feature importances in descending order
indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)

# Select the top k features (e.g., top 5)
k = 4
top_k_features = [features[i] for i in indices[:k]]

# Calculate the mean of the target variable (ranking)
target_mean = np.mean(df[target])

# Create an array of target variable means with the same length as the testing set
y_mean = np.full_like(y_test, target_mean)

# Calculate the squared sum of means
squared_sum_of_means = np.sum(np.square(y_test - y_mean))

st.title("Premier Ranking")

with st.container():
    image = Image.open('premierl.png')

    # Display the image with center alignment
    st.image(image, use_column_width=True)

    st.write("The Premier League (legal name: The Football Association Premier League Limited) is the highest level of the English football league system. Contested by 20 clubs, it operates on a system of promotion and relegation with the English Football League (EFL). Seasons typically run from August to May with each team playing 38 matches against all other teams both home and away. Most games are played on Saturday and Sunday afternoons, with occasional weekday evening fixtures. The competition was founded as the FA Premier League on 20 February 1992 following the decision of First Division (top-tier league from 1888 until 1992) clubs to break away from the English Football League. However, teams may still be relegated into and promoted from the EFL Championship. The Premier League takes advantage of a lucrative television rights sale to Sky: from 2019 to 2020, accumulated television rights were worth around £3.1 billion a year, with Sky and BT Group securing the domestic rights to broadcast 128 and 32 games respectively.")

# Create a dropdown box
selected_option = st.selectbox("Select a Team",
                               ["Arsenal", "Aston Villa", "Birmingham City", "Blackburn", "Blackpool", "Bolton",
                                "Bournemouth", "Brentford", "Brighton", "Burnley", "Cardiff City", "Chelsea", "Crystal Palace",
                                "Everton", "Fulham", "Huddersfield", "Hull City", "Leeds United", "Leicester City", "Liverpool",
                                "Manchester City", "Manchester Utd", "Middlesbrough", "Newcastle Utd", "Norwich City", "Nott'ham Forest",
                                "QPR", "Reading", "Sheffield Utd", "Southampton", "Stoke City", "Sunderland", "Swansea City", "Tottenham",
                                "Watford", "West Brom", "West Ham", "Wigan Athletic", "Wolves"])

# Display the selected option
st.write("You selected:", selected_option)

# Create a multiselect box
selected_options = st.multiselect('Select options', ['Pts', 'Pts/MP', 'GD', 'W'])

# Button for Rank prediction
result = st.button("PREDICT RANK")

# Handle button click event
if result:
    # Predict the ranking for the selected team
    team_features = df[df['Squad'] == selected_option][features]
    team_ranking = rf.predict(team_features)[0]

    # Display the predicted ranking
    st.write("Predicted Ranking:", team_ranking)

    # Display the top k features
    st.write(f'Top {k} features:', top_k_features)

    # Display the squared sum of means
    st.write("Squared Sum of Means:", squared_sum_of_means)


