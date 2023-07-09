import joblib
from flask import Flask, render_template, request
import pandas as pd

# Load the dataset
data = pd.read_csv('cleaned_data.csv')

# Extract location options from the dataset
locations = data['location'].unique()

# Create the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_prediction_model.joblib')

# Perform one-hot encoding
encoded_data = pd.get_dummies(data, columns=['location'])

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = encoded_data.drop('price', axis=1)  # Input features
y = encoded_data['price']  # Target variable
# Home page
@app.route('/')
def home():
    return render_template('index.html', locations=locations)

# Prediction result page
# Prediction result page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'location': [location],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'bhk': [bhk]
    })
    print(input_data)
    print(location," ",total_sqft," ",bath," ",bhk)

    # Example prediction
    new_data = pd.DataFrame({
        'location': [location],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'bhk': [bhk]
    })
    # Perform one-hot encoding for the new data
    encoded_new_data = pd.get_dummies(new_data, columns=['location'])

    # Align the new data with the training data columns
    encoded_new_data = encoded_new_data.reindex(columns=X.columns, fill_value=0)

    # Make predictions on the new data
    predicted_price = model.predict(encoded_new_data)

    print('Predicted Price:', predicted_price)

    # # Perform one-hot encoding for the input data
    # encoded_data = pd.get_dummies(input_data, columns=['location'])
    #
    # # Get the column names after one-hot encoding
    # encoded_columns = encoded_data.columns
    #
    # # Align the new data with the column names of the trained model
    # encoded_new_data = encoded_data.reindex(columns=encoded_columns, fill_value=0)



    # # Make the prediction using the loaded model
    # predicted_price = model.predict(encoded_new_data)

    # Render the prediction result page with the predicted price
    return render_template('result.html', predicted_price=predicted_price[0]*10000)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
