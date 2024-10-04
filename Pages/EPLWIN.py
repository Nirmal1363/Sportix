import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import json
import requests
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie

# Load the matches data
matches = pd.read_csv("matches_2.csv", index_col=0)

# Preprocessing code
matches["date"] = pd.to_datetime(matches["date"])
matches["target"] = (matches["result"] == "W").astype("int")
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek


# Rolling averages function
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

grouped_matches = matches.groupby("team")
matches_rolling = grouped_matches.apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Model training code
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = matches_rolling[matches_rolling["date"] < '2023-01-01']
test = matches_rolling[matches_rolling["date"] > '2023-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])
error = accuracy_score(test["target"], preds)

#Lottie files
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie animation files
lottie_coding1 = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_yLCHY6q3aU.json")
lottie_coding2 = load_lottiefile("Pages/versus.json")

# Streamlit web app code
st.title("Premier League")

# Display the image
image = Image.open('premierl.png')
st.image(image, use_column_width=True)

st.write(
    "The Premier League (legal name: The Football Association Premier League Limited) is the highest level of the English football league system. Contested by 20 clubs, it operates on a system of promotion and relegation with the English Football League (EFL). Seasons typically run from August to May with each team playing 38 matches against all other teams both home and away. Most games are played on Saturday and Sunday afternoons, with occasional weekday evening fixtures. The competition was founded as the FA Premier League on 20 February 1992 following the decision of First Division (top-tier league from 1888 until 1992) clubs to break away from the English Football League. However, teams may still be relegated into and promoted from the EFL Championship. The Premier League takes advantage of a lucrative television rights sale to Sky: from 2019 to 2020, accumulated television rights were worth around £3.1 billion a year, with Sky and BT Group securing the domestic rights to broadcast 128 and 32 games respectively.")

# Create a dropdown box for selecting Team 1
selected_team1 = st.selectbox("Select Team 1",
                              ["Brighton and Hove Albion", "Manchester United", "Newcastle United", "Tottenham Hotspur",
                               "West Ham United", "Wolverhampton Wanderers"])

# Styling with CSS
st.markdown(
    """
    <style>
    .lottie-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the Lottie animation
st.markdown("<div class='lottie-container'>", unsafe_allow_html=True)
st_lottie(lottie_coding2, height=200)
st.markdown("</div>", unsafe_allow_html=True)

# Create a dropdown box for selecting Team 2
selected_team2 = st.selectbox("Select Team 2",
                              ["Brighton and Hove Albion", "Manchester United", "Newcastle United", "Tottenham Hotspur",
                               "West Ham United", "Wolverhampton Wanderers"])

# Make predictions when the "Predict" button is clicked
if st.button("Predict"):
    with st.spinner("Predicting..."):
        # Filter the matches data for the selected teams
        team1_matches = matches_rolling[matches_rolling["team"] == selected_team1]
        team2_matches = matches_rolling[matches_rolling["team"] == selected_team2]

        # Combine the data for both teams
        data = pd.concat([team1_matches, team2_matches])

        # Make predictions
        preds = rf.predict(data[predictors])
        combined = pd.DataFrame(dict(actual=data["target"], predicted=preds), index=data.index)
        precision = precision_score(data["target"], preds)

        st.write("Precision Score:", precision)
        st.write(combined)


