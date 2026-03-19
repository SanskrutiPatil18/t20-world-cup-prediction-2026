import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model, scaler, and label encoder
try:
    best_xgb_model = joblib.load('best_xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Model components not found. Please make sure 'best_xgb_model.joblib', 'scaler.joblib', and 'label_encoder.joblib' are in the same directory.")
    st.stop()

st.set_page_config(layout='wide')
st.title('ICC Men’s T20 World Cup 2026 Winner Prediction')
st.write('Predict the winner of a T20 cricket match based on various team and venue metrics.')

# Input features from the user
st.header('Match Details')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Team A Metrics')
    team_a_ranking = st.slider('Team A Ranking', 1, 20, 5)
    team_a_form = st.slider('Team A Form (0-100)', 0.0, 100.0, 75.0, 0.1)
    head_to_head_a_wins = st.slider('Head-to-Head A Wins', 0, 15, 2)
    venue_home_advantage_a = st.selectbox('Venue Home Advantage A', [0, 1])
    team_a_tech_index = st.slider('Team A Tech Index', 50.0, 300.0, 200.0, 0.1)

with col2:
    st.subheader('Team B Metrics')
    team_b_ranking = st.slider('Team B Ranking', 1, 20, 8)
    team_b_form = st.slider('Team B Form (0-100)', 0.0, 100.0, 60.0, 0.1)
    head_to_head_b_wins = st.slider('Head-to-Head B Wins', 0, 15, 3)
    venue_home_advantage_b = st.selectbox('Venue Home Advantage B', [0, 1])
    team_b_tech_index = st.slider('Team B Tech Index', 50.0, 300.0, 180.0, 0.1)

# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'Team_A_Ranking': team_a_ranking,
    'Team_B_Ranking': team_b_ranking,
    'Team_A_Form': team_a_form,
    'Team_B_Form': team_b_form,
    'HeadToHead_A_Wins': head_to_head_a_wins,
    'HeadToHead_B_Wins': head_to_head_b_wins,
    'Venue_HomeAdvantage_A': venue_home_advantage_a,
    'Venue_HomeAdvantage_B': venue_home_advantage_b,
    'Team_A_Tech_Index': team_a_tech_index,
    'Team_B_Tech_Index': team_b_tech_index
}])

# Scale the input data
scaled_input = scaler.transform(input_data)

if st.button('Predict Winner'):
    prediction_encoded = best_xgb_model.predict(scaled_input)
    predicted_winner = label_encoder.inverse_transform(prediction_encoded)
    
    st.success(f'The predicted winner is: **{predicted_winner[0]}**')
    
    # Display feature importances (optional, for insights)
    st.subheader('Feature Importance for Prediction')
    feature_importances = best_xgb_model.feature_importances_
    feature_names = input_data.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))
