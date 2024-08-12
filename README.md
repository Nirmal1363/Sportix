## Project Overview
This project aims to predict the winners and losers of NBA games and outcomes of EPL matches using machine learning algorithms. We utilize Ridge classifier and RNN for NBA game predictions, and Random Forest for EPL game predictions.

## Abstract
The project develops and evaluates machine learning models for predicting NBA game outcomes and EPL match results. It utilizes Ridge classifier, RNN, and Random Forest algorithms to analyze historical data and train predictive models. The implemented models demonstrated promising accuracy in predicting winners/losers of NBA games and outcomes of EPL matches.

## Business Use Case
The project provides valuable predictive insights for the sports industry, benefiting sports analysts, fantasy sports platforms, and sports betting companies. It enables data-driven decision-making, enhances user experiences, and generates potential revenue streams in the sports industry.

## Concepts Used
- Data Extraction (Web Scraping using BeautifulSoup)
- Data Pre-Processing
- Feature Engineering
- Data Visualization
- Model Building (Ridge Classification, Random Forest, XGBoost Model)
- Model Optimization/Fine Tuning

## Data
Three datasets were used:
1. NBA game-by-game box scores (17772 rows, 145 attributes)
2. EPL games from 2022 and 2023 seasons (1389 rows, 26 attributes)
3. EPL final points table from 2010-2023 (260 rows, 15 attributes)

## Models Used
- NBA Winning Prediction: Ridge Classification, XGBoost Model, Random Forest
- EPL Winning Prediction: LSTM, Random Forest
- EPL Ranking Prediction: Decision Tree, Logistic Regression, Random Forest

## Results
Comparison of models for NBA prediction:
- XGBoost: Accuracy - 0.9192, Precision - 0.9392, Recall - 0.8994, F1 Score - 0.9188
- Random Forest: Accuracy - 0.9970, Precision - 0.9981, Recall - 0.9959, F1 Score - 0.9970

## Future Enhancements
1. Adding more dataset
2. Sentiment Analysis and Social Media Data
3. Player-Specific Data and Biometrics
4. Integration of Expert Insights
5. Enhanced Data Visualization

## Deployment
The ML model has been integrated into a Streamlit web app for real-time predictions. The app can be accessed at: http://10.120.8.201:8501
