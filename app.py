from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__ )

# Load the trained Random Forest Classifier model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the user through a form
        tempp = float(request.form['tempp'])
        co2 = float(request.form['co2'])
        light = float(request.form['light'])

        # Prepare the input data for prediction
        input_data = np.array([[tempp, light, co2]])

        # Make the prediction using the loaded model
        prediction = model.predict(input_data)

        # Map the prediction to human-readable labels
        result = "Room Occupied" if prediction[0] == 1 else "Room Not Occupied"

        # Return the predicted class as a response
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port = 3000,debug=True)