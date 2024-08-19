# Flight Cancellation Prediction

## Overview

This project involves predicting flight cancellations using a dataset containing flight-related features. The dataset consists of 100,000 entries with 40 columns, covering various aspects of flight data such as departure time, carrier information, weather conditions, and more.

## Dataset

The dataset has the following columns:

- **FL_DATE**: Flight date
- **DEP_HOUR**: Departure hour
- **MKT_UNIQUE_CARRIER**: Marketing carrier
- **MKT_CARRIER_FL_NUM**: Marketing carrier flight number
- **OP_UNIQUE_CARRIER**: Operating carrier
- **OP_CARRIER_FL_NUM**: Operating carrier flight number
- **TAIL_NUM**: Tail number
- **ORIGIN**: Origin airport code
- **DEST**: Destination airport code
- **DEP_TIME**: Departure time
- **CRS_DEP_TIME**: Scheduled departure time
- **TAXI_OUT**: Taxi out time
- **DEP_DELAY**: Departure delay
- **AIR_TIME**: Air time
- **DISTANCE**: Flight distance
- **CANCELLED**: Cancellation indicator
- **LATITUDE**: Latitude of origin airport
- **LONGITUDE**: Longitude of origin airport
- **ELEVATION**: Elevation of origin airport
- **MESONET_STATION**: Mesonet station
- **YEAR OF MANUFACTURE**: Year of aircraft manufacture
- **MANUFACTURER**: Aircraft manufacturer
- **ICAO TYPE**: ICAO aircraft type
- **RANGE**: Aircraft range
- **WIDTH**: Aircraft width
- **WIND_DIR**: Wind direction
- **WIND_SPD**: Wind speed
- **WIND_GUST**: Wind gust
- **VISIBILITY**: Visibility
- **TEMPERATURE**: Temperature
- **DEW_POINT**: Dew point
- **REL_HUMIDITY**: Relative humidity
- **ALTIMETER**: Altimeter setting
- **LOWEST_CLOUD_LAYER**: Lowest cloud layer
- **N_CLOUD_LAYER**: Number of cloud layers
- **LOW_LEVEL_CLOUD**: Low-level cloud
- **MID_LEVEL_CLOUD**: Mid-level cloud
- **HIGH_LEVEL_CLOUD**: High-level cloud
- **CLOUD_COVER**: Cloud cover
- **ACTIVE_WEATHER**: Active weather conditions

## Exploratory Data Analysis (EDA)

A detailed EDA was performed to understand the dataset, identify patterns, and visualize relationships between features. Key insights from the EDA were used to guide model selection and hyperparameter tuning.

## Model Building and Evaluation

### Models Used

1. **Random Forest**
2. **Logistic Regression**
3. **Decision Tree**

GridSearchCV was utilized to optimize hyperparameters for the Random Forest, Logistic Regression, and Decision Tree models. The Random Forest model achieved the highest accuracy among the models evaluated.

### Random Forest Model

- **Model**: Random Forest
- **Accuracy**: [Insert accuracy here]

### TensorFlow Keras Model

A TensorFlow Keras model was also built for comparison. This model's performance was compared with the Random Forest model.

## Streamlit Applications

Two Streamlit applications were developed:

1. **Flight Cancellation Prediction with Random Forest**: This application allows users to input flight details and receive a prediction on whether the flight will be canceled or not.

2. **Flight Cancellation Prediction with TensorFlow Keras**: This application provides similar functionality but uses a TensorFlow Keras model for predictions.

## Installation

To set up the environment, you need to install the following packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow streamlit
