pip install pandas streamlit numpy pydeck requests scikit-learn matplotlib seaborn

import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
import pickle
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='XGB Regression', page_icon=':money_with_wings:')

@st.cache_data
def load_data():
    # First load the original airbnb listtings dataset
    data = pd.read_csv("listings.csv") 
    return data

data = load_data()
st.sidebar.title("Exploring Listing Price with Regression Model")
st.sidebar.markdown("This web app allows you to explore eXtreme Gradient Boosting Regression for the prediction of Airbnb Prices. You can view the distribution of price (target) on a visualization in the 'Explore' tab and make predictions in the 'Predict' tab.")

min_nights_range = st.sidebar.slider("Minimum Nights Range", float(data['minimum_nights'].min()), float(data['minimum_nights'].max()), (float(data['minimum_nights'].min()), float(data['minimum_nights'].max())))
num_reviews_range = st.sidebar.slider("Number of Reviews Range", float(data['number_of_reviews'].min()), float(data['number_of_reviews'].max()), (float(data['number_of_reviews'].min()), float(data['number_of_reviews'].max())))
reviews_per_month_range = st.sidebar.slider("Reviews per Month Range", float(data['reviews_per_month'].min()), float(data['reviews_per_month'].max()), (float(data['reviews_per_month'].min()), float(data['reviews_per_month'].max())), 0.01)
host_listings_count_range = st.sidebar.slider("Host Listings Count Range", float(data['calculated_host_listings_count'].min()), float(data['calculated_host_listings_count'].max()), (float(data['calculated_host_listings_count'].min()), float(data['calculated_host_listings_count'].max())))
availability_range = st.sidebar.slider("Availability Range", float(data['availability_365'].min()), float(data['availability_365'].max()), (float(data['availability_365'].min()), float(data['availability_365'].max())))

neighbourhood = st.sidebar.selectbox("Neighbourhood", ['Bukit Timah', 'Bukit Merah', 'Newton', 'Geylang', 'River Valley',
                                    'Rochor', 'Queenstown', 'Marine Parade', 'Toa Payoh', 'Outram',
                                    'Tanglin', 'Kallang', 'Novena', 'Downtown Core', 'Singapore River',
                                    'Orchard', 'Others'], key='neighbourhood')


room_type = st.sidebar.selectbox("Room Type", ['Private room', 'Entire home/apt', 'Shared room'], key='room_type')
    
    
tab1, tab2 = st.tabs(['Explore', 'Predict'])

with tab1:
    filtered_data = data[
        (data['minimum_nights'] >= min_nights_range[0]) & (data['minimum_nights'] <= min_nights_range[1]) &
        (data['number_of_reviews'] >= num_reviews_range[0]) & (data['number_of_reviews'] <= num_reviews_range[1]) &
        (data['reviews_per_month'] >= reviews_per_month_range[0]) & (data['reviews_per_month'] <= reviews_per_month_range[1]) &
        (data['calculated_host_listings_count'] >= host_listings_count_range[0]) & (data['calculated_host_listings_count'] <= host_listings_count_range[1]) &
        (data['availability_365'] >= availability_range[0]) & (data['availability_365'] <= availability_range[1])
    ]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Set Seaborn style to darkgrid
    sns.set(style="darkgrid")

    # Create a histogram using Seaborn
    sns.histplot(filtered_data['price'], bins=20, kde=False, color='red', element='step', stat='density')

    # Customize labels and title
    plt.xlabel('Price', color='white')
    plt.ylabel('Probability Density Function')
    plt.title('Price Distribution')

    # Show the plot
    st.pyplot(fig)


# Tab 2: Predict the Price using XGBoost Model
with tab2:
    # Load the serialized trained model XGBRegmodel.pkl
    with open('XGBRegmodel.pkl', 'rb') as file:
        rf = pickle.load(file)

    # Define the app title and favicon
    st.title('Prediction of Price - How valuable is your listing?') 
    st.subheader('Predict')
    st.markdown("This tab allows you to make predictions on the price based on the input variables.")
    
    # Define neighborhood options
    neighborhood_options = ['Bukit Timah', 'Bukit Merah', 'Newton', 'Geylang', 'River Valley',
                            'Rochor', 'Queenstown', 'Marine Parade', 'Toa Payoh', 'Outram',
                            'Tanglin', 'Kallang', 'Novena', 'Downtown Core', 'Singapore River',
                            'Orchard', 'Others']

    # Input widgets for user input
    pred_min_nights = st.number_input("Minimum Nights", int(data['minimum_nights'].min()), int(data['minimum_nights'].max()), int(data['minimum_nights'].mean()), 1)
    pred_num_reviews = st.number_input("Number of Reviews", int(data['number_of_reviews'].min()), int(data['number_of_reviews'].max()), int(data['number_of_reviews'].mean()), 1)
    pred_reviews_per_month = st.number_input("Reviews per Month", int(data['reviews_per_month'].min()), int(data['reviews_per_month'].max()), int(data['reviews_per_month'].mean()), 1)
    pred_host_listings_count = st.number_input("Host Listings Count", int(data['calculated_host_listings_count'].min()), int(data['calculated_host_listings_count'].max()), int(data['calculated_host_listings_count'].mean()), 1)
    pred_availability = st.number_input("Availability", int(data['availability_365'].min()), int(data['availability_365'].max()), int(data['availability_365'].mean()), 1)
    pred_room_type = st.selectbox("Room Type", data['room_type'].unique())
    pred_neighbourhood = st.selectbox("Neighbourhood", options=neighborhood_options)

    # Additional features
    pred_mrt = st.number_input("Listing name contains : MRT? (1 - yes, 0 - no)", min_value=0, max_value=1)
    pred_private = st.number_input("Listing name contains : Private? (1 - yes, 0 - no)", min_value=0, max_value=1)
    pred_spacious = st.number_input("Listing name contains : Spacious? (1 - yes, 0 - no)", min_value=0, max_value=1)

    # Last review date features
    pred_last_review_year = st.number_input("Last Review Year", min_value=2013, max_value=2019, value=2016)
    pred_last_review_month = st.number_input("Last Review Month", min_value=1, max_value=12, value=6)
    pred_last_review_day = st.number_input("Last Review Day", min_value=1, max_value=31, value=15)

    # Host ID
    pred_host_id = st.number_input("Host ID", value=int(data['host_id'].mean()))

    # Define a dictionary (neighborhood_mapping) that maps neighborhood to their corresponding integer values.
    neighborhood_mapping = {
        'Kallang': 1043,
        'Geylang': 994,
        'Novena': 537,
        'Rochor': 536,
        'Outram': 477,
        'Bukit Merah': 469,
        'Downtown Core': 428,
        'River Valley': 362,
        'Queenstown': 266,
        'Tanglin': 210,
        'Singapore River': 175,
        'Marine Parade': 171,
        'Others': 137,
        'Orchard': 136,
        'Newton': 134,
        'Bukit Timah': 131,
        'Toa Payoh': 101
    }

    room_mapping = {'Entire home/apt': 2, 'Private room': 1, 'Shared room': 0}

    # Create a function that takes neighbourhood as an argument and returns the corresponding integer value.
    def match_neighbourhood(pred_neighbourhood):
        return neighborhood_mapping[pred_neighbourhood]

    def match_room_type(pred_room_type):
        return room_mapping[pred_room_type]

    # Create a price prediction button
    if st.button('Predict price'):
        # Call the function with the selected room_type as an argument
        pred_neighbourhood_numeric = match_neighbourhood(pred_neighbourhood)
        pred_room_type_numeric = match_room_type(pred_room_type)

        # Normalization based on your provided formula
        pred_min_nights_normalized = (pred_min_nights - data['minimum_nights'].mean()) / data['minimum_nights'].std()
        pred_num_reviews_normalized = (pred_num_reviews - data['number_of_reviews'].mean()) / data['number_of_reviews'].std()
        pred_reviews_per_month_normalized = (pred_reviews_per_month - data['reviews_per_month'].mean()) / data['reviews_per_month'].std()
        pred_host_listings_count_normalized = (pred_host_listings_count - data['calculated_host_listings_count'].mean()) / data['calculated_host_listings_count'].std()
        pred_availability_normalized = (pred_availability - data['availability_365'].mean()) / data['availability_365'].std()

        # Make the prediction   
        input_data = [[pred_host_id, pred_neighbourhood_numeric, pred_room_type_numeric, pred_min_nights_normalized,
                       pred_num_reviews_normalized, pred_reviews_per_month_normalized,
                       pred_host_listings_count_normalized, pred_availability_normalized,
                       pred_last_review_year, pred_last_review_month, pred_last_review_day,
                       pred_mrt, pred_private, pred_spacious]]

        # Update the column names in the input_df
        input_df = pd.DataFrame(input_data, columns=['host_id', 'neighbourhood', 'room_type', 'minimum_nights',
                                             'number_of_reviews', 'reviews_per_month',
                                             'calculated_host_listings_count', 'availability_365',
                                             'last_review_year', 'last_review_month', 'last_review_day',
                                             'mrt', 'private', 'spacious'])

        predicted_price = rf.predict(input_df)[0]

        # Display the price prediction
        st.write('## Prediction Result')
        st.write(f'**Predicted Price:** ${predicted_price:,.2f}', font=("Helvetica", 18), color="green")



