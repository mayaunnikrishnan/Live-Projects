import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model # type: ignore
from sklearn.pipeline import Pipeline
import numpy as np

# Load the trained model and frequency-encoded values
#model = open("trained_model.pkl", "rb")
#rf = joblib.load(model)

frequency_path = open("frequency_encoded_values.pkl", "rb")
frequency_encoded_values = joblib.load(frequency_path)

# Load the saved model
model = load_model('ann_model.h5')


# Create input fields for user input
st.title("FLIGHT CANCELLATION PREDICTION PROJECT APP")
#Create input fields for user input
# Create input fields for user input
DEP_HOUR = st.sidebar.slider("Departure Hour", 0, 23, step=1)
DISTANCE = st.sidebar.number_input("Distance (miles)")
LATITUDE = st.sidebar.slider("Latitude", 18.0, 64.0, step=1.0)
LONGITUDE = st.sidebar.slider("Longitude", -150.0, -50.0, step=1.0)
TEMPERATURE = st.sidebar.slider("Temperature", -20.0, 42.0, step=1.0)
DEW_POINT = st.sidebar.slider("Dew Point", -20.0, 30.0, step=1.0)
REL_HUMIDITY = st.sidebar.slider("Relative Humidity", 2.0, 100.0, step=1.0)
LOWEST_CLOUD_LAYER = st.sidebar.slider("Lowest Cloud Layer", 0.0, 33000.0, step=100.0)
MONTH = st.sidebar.slider("Month", 1, 12, step=1)
WIND_SPD = st.sidebar.slider("Wind Speed", 0.0, 32.0, step=1.0)
WIND_GUST = st.sidebar.slider("Wind Gust", 0.0, 55.0, step=1.0)
VISIBILITY = st.sidebar.slider("Visibility", 0.0, 10.0, step=1.0)
ALTIMETER = st.sidebar.slider("Atmospheric Pressure", 25.0, 31.0, step=0.1)
ICAO_TYPE = st.sidebar.selectbox("ICAO Type", ['A321', 'B39M', 'E75L', 'B737',
                                              'A320', 'A319', 'B38M', 'B738', 'CRJ2', 'A20N', 'CRJ7',
                                              'B712', 'E145', 'CRJ9', 'B739', 'B77W', 'A21N',
                                              'B752', 'B764', 'E170', 'E190', 'BCS3', 'B772', 'BCS1',
                                              'B789', 'DH8D', 'A332', 'B753', 'B763', 'B78X', 'A333',
                                              'B788', 'A339', 'A359'])
ORIGIN_STATE_NAME = st.sidebar.selectbox("Origin State Name", ['Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California', 'Colorado',
                                                               'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana',
                                                               'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', 'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri',
                                                               'Mississippi', 'Montana', 'North Carolina', 'North Dakota', 'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico',
                                                               'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island',
                                                               'South Carolina', 'South Dakota', 'Tennessee', 'U.S. Pacific Trust Territories and Possessions',
                                                               'Texas', 'Utah', 'Virginia', 'U.S. Virgin Islands', 'Vermont', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming'])
RANGE = st.sidebar.selectbox("Range", ["Short Range", "Medium Range", "Long Range"])
WIDTH = st.sidebar.selectbox("Width", ["Narrow-body", "Wide-body"])
ACTIVE_WEATHER = st.sidebar.selectbox("Active Weather", [0.0, 1.0, 2.0])
CLOUD_COVER = st.sidebar.selectbox("Cloud Cover", [1.0, 2.0, 3.0, 4.0, 0.0])
N_CLOUD_LAYER = st.sidebar.selectbox("N_Cloud Layer", [1.0, 3.0, 0.0, 2.0, 4.0])


# Create and fit LabelEncoders for categorical features
LE_ICAO_TYPE = LabelEncoder()
LE_ICAO_TYPE.fit( ['A321', 'B39M' ,'E75L', 'B737' ,
                                        'A320', 'A319' ,'B38M', 'B738' ,'CRJ2', 'A20N','CRJ7',
                                        'B712' ,'E145' ,'CRJ9', 'B739', 'B77W' ,'A21N',
                                        'B752' ,'B764' ,'E170','E190' ,'BCS3', 'B772', 'BCS1',
                                        'B789', 'DH8D', 'A332' ,'B753' ,'B763' ,'B78X','A333',
                                        'B788', 'A339', 'A359'])

LE_ORIGIN_STATE_NAME = LabelEncoder()
LE_ORIGIN_STATE_NAME.fit(['Alaska', 'Alabama',
                                    'Arkansas', 'Arizona', 'California','Colorado',
                                    'Connecticut', 'Delaware', 'Florida', 'Georgia',
                                    'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana',
                                    'Kansas','Kentucky', 'Louisiana', 'Massachusetts',
                                    'Maryland', 'Maine','Michigan', 'Minnesota', 'Missouri',
                                    'Mississippi', 'Montana','North Carolina', 'North Dakota',
                                    'Nebraska', 'New Hampshire','New Jersey', 'New Mexico',
                                    'Nevada', 'New York', 'Ohio','Oklahoma', 'Oregon',
                                    'Pennsylvania', 'Puerto Rico','Rhode Island',
                                    'South Carolina', 'South Dakota', 'Tennessee',
                                    'U.S. Pacific Trust Territories and Possessions',
                                    'Texas', 'Utah','Virginia', 'U.S. Virgin Islands',
                                    'Vermont', 'Washington','Wisconsin', 'West Virginia',
                                    'Wyoming'])  # List all values from ORIGIN_STATE_NAME

# Create a function to predict cancellation
def predict_cancellation(DEP_HOUR, DISTANCE, LATITUDE, LONGITUDE, ICAO_TYPE_encoded,
                          WIND_SPD, WIND_GUST, VISIBILITY, TEMPERATURE, DEW_POINT,
                          REL_HUMIDITY, ALTIMETER, LOWEST_CLOUD_LAYER, N_CLOUD_LAYER,
                          CLOUD_COVER, ACTIVE_WEATHER, MONTH, ORIGIN_STATE_NAME_encoded,
                          RANGE, WIDTH):
    ICAO_TYPE_encoded = LE_ICAO_TYPE.transform([ICAO_TYPE])[0]
    ORIGIN_STATE_NAME_encoded = LE_ORIGIN_STATE_NAME.transform([ORIGIN_STATE_NAME])[0]
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'DEP_HOUR': [DEP_HOUR],
        'DISTANCE': [DISTANCE],
        'LATITUDE': [LATITUDE],
        'LONGITUDE': [LONGITUDE],
        'ICAO_TYPE': [ICAO_TYPE_encoded],
        'WIND_SPD': [WIND_SPD],
        'WIND_GUST': [WIND_GUST],
        'VISIBILITY': [VISIBILITY],
        'TEMPERATURE': [TEMPERATURE],
        'DEW_POINT': [DEW_POINT],
        'REL_HUMIDITY': [REL_HUMIDITY],
        'ALTIMETER': [ALTIMETER],
        'LOWEST_CLOUD_LAYER': [LOWEST_CLOUD_LAYER],
        'N_CLOUD_LAYER': [N_CLOUD_LAYER],
        'CLOUD_COVER': [CLOUD_COVER],
        'ACTIVE_WEATHER': [ACTIVE_WEATHER],
        'MONTH': [MONTH],
        'ORIGIN_STATE_NAME': [ORIGIN_STATE_NAME_encoded],
        'RANGE': [RANGE],
        'WIDTH': [WIDTH]
    })

    # Apply frequency encoding to RANGE and WIDTH
    input_data['RANGE'] = input_data['RANGE'].map(frequency_encoded_values['RANGE'])
    input_data['WIDTH'] = input_data['WIDTH'].map(frequency_encoded_values['WIDTH'])

    # Scale the input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make a prediction using the ANN model
    prediction = model.predict(input_data_scaled)
    prediction_class = np.argmax(prediction[0])

    return prediction_class

# Create a prediction button
prediction_button = st.sidebar.button("Predict Cancellation")

# Perform prediction when the button is clicked
if prediction_button:
    prediction = predict_cancellation(DEP_HOUR, DISTANCE, LATITUDE, LONGITUDE, ICAO_TYPE,
                                      WIND_SPD, WIND_GUST, VISIBILITY, TEMPERATURE,
                                      DEW_POINT, REL_HUMIDITY, ALTIMETER,
                                      LOWEST_CLOUD_LAYER, N_CLOUD_LAYER, CLOUD_COVER,
                                      ACTIVE_WEATHER, MONTH, ORIGIN_STATE_NAME,
                                      RANGE, WIDTH)

    # Display the prediction
    if prediction == 0:
        st.write('Flight Not Cancelled', font=('Arial', 24, 'bold'))
        st.image("not_cancelled_image.jpg")  # Display image for "Not Cancelled"
    elif prediction == 1:
        st.write('Carrier Cancellation', font=('Arial', 24, 'bold'))
        st.image("Carrier_cancelled_image.jpg")  # Display image for "Carrier Cancelled"
    elif prediction == 2:
        st.write('Weather Cancellation', font=('Arial', 24, 'bold'))
        st.image("Weather_cancelled_image.jpg")  # Display image for "Weather Cancelled"
    elif prediction == 3:
        st.write('National Air System Cancellation', font=('Arial', 24, 'bold'))
        st.image("National_cancelled_image.jpg")  # Display image for "National Air System Cancelled"
    else:
        st.write('Security Cancellation', font=('Arial', 24, 'bold'))
        st.image("Security_cancelled_image.jpg")  # Display image for "Security Cancelled"