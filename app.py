from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('dbfinal.csv', parse_dates=['Date'])

# Group data by 'States/UTs'
grouped_data = df.groupby('States/UTs')

# Dictionary to store models and predictions for each state
state_models = {}

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 7
all_states_predictions_df = pd.DataFrame()

# Loop over each state
for state, state_df in grouped_data:
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    state_prices_normalized = scaler.fit_transform(np.array(state_df['Tomato']).reshape(-1, 1))

    # Prepare the data for LSTM
    X, Y = create_dataset(state_prices_normalized, look_back)

    # Reshape the input data for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, Y, epochs=100, batch_size=32, verbose=1)

    # Save the model in the dictionary for future use
    state_models[state] = model

# Function to get predictions for a specific state
def get_state_predictions(state):
    last_sequence = state_prices_normalized[-look_back:]
    future_predictions = []

    for _ in range(30):
        next_day_prediction = state_models[state].predict(np.reshape(last_sequence, (1, look_back, 1)))
        future_predictions.append(next_day_prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], next_day_prediction[0, 0])

    # Inverse transform the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Create a DataFrame for future predictions for the current state
    future_predictions_df = pd.DataFrame({
        'Date': pd.date_range(start=state_df['Date'].max() + pd.DateOffset(1), periods=30),
        'Predicted_Price': future_predictions.flatten(),
        'State': state  # Adding a column to store the state name
    })

    default_value = state_prices_normalized[-1][0]
    future_predictions_df['Predicted_Price'].fillna(default_value, inplace=True)

    return future_predictions_df

@app.route('/')
def index():
    states = sorted(df['States/UTs'].unique())
    return render_template('pr1.html', predictions=[], states=states, selected_state=None)

@app.route('/predict', methods=['POST'])
def predict():
    selected_state = request.form.get('state')
    if selected_state not in state_models:
        return "Invalid state selection."

    state_prediction_df = get_state_predictions(selected_state)
    states = sorted(df['States/UTs'].unique())

    return render_template('pr1.html', predictions=state_prediction_df.to_dict(orient='records'), selected_state=selected_state, states=states)

if __name__ == '__main__':
    app.run(debug=True)
