import pandas as pd
import streamlit as st
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder

# Load Airbnb data
@st.cache_data
def load_data():
    data = pd.read_csv("listings.csv")  # Update with your actual CSV file
    return data

data = load_data()

# Sidebar setup
st.sidebar.title("Airbnb Listing Price Prediction")
st.sidebar.markdown("This web app allows you to predict Airbnb listing prices using an XGBoost regression model. You can input features such as neighborhood, room type, minimum nights, etc.")

# Feature sliders and selectors
neighborhoods = data['neighbourhood'].unique()
selected_neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods)

room_types = data['room_type'].unique()
selected_room_type = st.sidebar.selectbox("Room Type", room_types)

minimum_nights = st.sidebar.slider("Minimum Nights", 1, 30, 3)

number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 100, 10)

calculated_host_listings_count = st.sidebar.slider("Host Listings Count", 1, 50, 5)

availability_365 = st.sidebar.slider("Availability (in days)", 0, 365, 30)

# Load the XGBoost model
with open('XGBRegModel.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Transform categorical features
le = LabelEncoder()
data['neighbourhood'] = le.fit_transform(data['neighbourhood'])
data['room_type'] = le.fit_transform(data['room_type'])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'neighbourhood': [le.transform([selected_neighborhood])[0]],
    'room_type': [le.transform([selected_room_type])[0]],
    'minimum_nights': [minimum_nights],
    'number_of_reviews': [number_of_reviews],
    'calculated_host_listings_count': [calculated_host_listings_count],
    'availability_365': [availability_365]
})

# Prediction
predicted_price = xgb_model.predict(input_data)[0]

# Main content area
st.title('Airbnb Listing Price Prediction')
st.subheader('Predict')

# Display input features
st.write('Selected Neighborhood:', selected_neighborhood)
st.write('Selected Room Type:', selected_room_type)
st.write('Minimum Nights:', minimum_nights)
st.write('Number of Reviews:', number_of_reviews)
st.write('Host Listings Count:', calculated_host_listings_count)
st.write('Availability (in days):', availability_365)

# Display prediction
st.write('The predicted Airbnb listing price is $', round(predicted_price, 2))
