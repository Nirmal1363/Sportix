import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.tree import export_text
import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie

# Load NBA data and perform preprocessing
df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date")
df = df.reset_index(drop=True)
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

# Function to add the target variable
def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

# Apply target creation per team
df = df.groupby("team", group_keys=False).apply(add_target)

# Handle nulls in the target variable
df.loc[pd.isnull(df["target"]), "target"] = 2  # Handling nulls
df["target"] = df["target"].astype(int, errors="ignore")

# Drop columns with null values
nulls = pd.isnull(df).sum()
nulls = nulls[nulls > 0]
valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns].copy()

# Define features and drop columns not needed for training
features = df.drop(['won', 'date'], axis=1)

# Encode categorical variables
categorical_features = ['team', 'team_opp']
encoder = OneHotEncoder(sparse_output=False)  # Updated parameter
encoded_features = encoder.fit_transform(features[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_features = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Concatenate the encoded features with the remaining features
features = pd.concat([features.drop(categorical_features, axis=1), encoded_features], axis=1)

# Select the target variable
target = df['won']

# Feature selection using SelectKBest
feature_selector = SelectKBest(score_func=f_classif, k=10)
selected_features = feature_selector.fit_transform(features, target)
selected_feature_names = features.columns[feature_selector.get_support()]
selected_features = pd.DataFrame(selected_features, columns=selected_feature_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.3, random_state=42)

# Create the random forest classifier
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Function to predict NBA winner and evaluation metrics
def predict_nba_winner(team1, team2):
    # Create a new DataFrame with the teams to predict
    teams_to_predict = pd.DataFrame([[team1, team2]], columns=['team', 'team_opp'])

    # Perform one-hot encoding for the teams to predict
    encoded_teams = encoder.transform(teams_to_predict[categorical_features])

    # Pad encoded_teams with zeros to match the number of features in the training data
    encoded_teams = np.pad(encoded_teams, ((0, 0), (0, features.shape[1] - encoded_teams.shape[1])), mode='constant')

    # Perform feature selection for the teams to predict
    selected_encoded_teams = feature_selector.transform(encoded_teams)
    selected_encoded_teams = pd.DataFrame(selected_encoded_teams, columns=selected_feature_names)

    # Pad selected_encoded_teams with zeros to match the number of features in the training data
    selected_encoded_teams = np.pad(selected_encoded_teams, ((0, 0), (0, selected_features.shape[1] - selected_encoded_teams.shape[1])), mode='constant')

    # Make predictions for the teams
    predictions = clf.predict(selected_encoded_teams)

    # Get the predicted winner
    if predictions[0] == 1:
        winner = team1
    else:
        winner = team2

    return winner

# Function to calculate evaluation metrics
def calculate_metrics():
    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc

# Function to generate decision tree text representation
def generate_decision_tree_text():
    # Get the first decision tree from the random forest
    tree = clf.estimators_[0]

    # Generate the text representation of the decision tree
    tree_text = export_text(tree, feature_names=selected_features.columns.tolist())

    return tree_text

# Streamlit web app
st.title("NBA Win Predictor")

# Lottie files
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie animation assets
lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_ef3beoqa.json")
lottie_coding2 = load_lottiefile("Pages/versus.json")

with st.container():
    # Display the NBA introduction and animation
    st_lottie(lottie_coding, height=350, key="coding")
    st.write("The National Basketball Association (NBA) is a professional basketball league in North America composed of 30 teams (29 in the United States and 1 in Canada)...")

# Create a dropdown box for Team 1 selection
selected_option1 = st.selectbox("Select Team 1", ["ATL", "DAL", "MIN", "SAS", "ORL", "MEM", "LAC", "DET", "MIL", "LAL"])

# Styling by CSS
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

# Create a dropdown box for Team 2 selection
selected_option2 = st.selectbox("Select Team 2", ["BOS", "BKN", "GSW", "PHI", "MIA", "TOR", "DEN", "UTA", "OKC", "NOP"])

# Add a button to predict the winner
if st.button("Predict Winner"):
    # Predict the winner
    winner = predict_nba_winner(selected_option1, selected_option2)

    # Display the predicted winner
    st.write("The predicted winner is: ", winner)

# Add a button to calculate evaluation metrics
if st.button("Calculate Metrics"):
    # Calculate evaluation metrics
    accuracy, precision, recall, f1, roc_auc = calculate_metrics()

    # Display the evaluation metrics
    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)
    st.write("ROC AUC Score:", roc_auc)

# Add a button to generate decision tree text representation
if st.button("Generate Decision Tree"):
    # Generate the decision tree text representation
    tree_text = generate_decision_tree_text()

    # Display the decision tree text representation
    st.code(tree_text)
