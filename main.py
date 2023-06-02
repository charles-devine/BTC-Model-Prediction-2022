import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("BTC-2021min.csv")

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Filter the data for the year 2022
data_2022 = data[data['date'].dt.year == 2022]

# Prepare the training data
X_train = data_2022[['unix', 'Volume BTC', 'Volume USD']]
y_train = data_2022['close']

# Create and train the random forest regression model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Generate the timestamps for 2022
timestamps_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='1min')

# Prepare the test data
X_test = pd.DataFrame({'unix': timestamps_2022.astype('int64'), 'Volume BTC': 0, 'Volume USD': 0})

# Generate the predictions for 2022
y_pred = model.predict(X_test)

# Create a new dataset with predicted prices for 2022
predictions_2022 = pd.DataFrame({'date': timestamps_2022, 'predicted_price': y_pred})
predictions_2022.to_csv('bitcoin_predictions_2022.csv', index=False)

# Print the predictions
print(predictions_2022)



