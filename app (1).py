import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Ensure the columns are in the same order as during training and handle categorical features
        # This assumes the input JSON has keys matching the original DataFrame columns (except 'Heart Disease')
        # and that one-hot encoding is handled correctly based on the training data's columns.
        # A more robust approach would involve saving the columns from the training data.

        # For demonstration, let's assume the columns are in a specific order and handle one-hot encoding
        # based on the original columns of the training data.
        # You might need to adjust this part based on the exact structure of your training data after get_dummies.

        # Example of how to handle one-hot encoding for new data:
        # Identify categorical columns from the original df (before get_dummies)
        original_categorical_cols = df.select_dtypes(include='object').columns.tolist()
        # Apply one-hot encoding to the new data
        df_processed = pd.get_dummies(df, columns=original_categorical_cols, drop_first=True)

        # Reindex the new data to match the columns of the training data (X_train)
        # This is crucial to avoid errors during prediction due to missing columns
        # You would ideally save the columns of X_train after training and use them here.
        # For this example, we'll get the columns from the current X variable after get_dummies.
        # In a real application, load the saved column list.
        train_columns = X.columns # Assuming X still holds the columns after get_dummies from the training phase
        df_processed = df_processed.reindex(columns=train_columns, fill_value=0)


        # Make prediction
        prediction = model.predict(df_processed)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI
    app.run(debug=True)
