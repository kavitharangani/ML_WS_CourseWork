from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("models/lr_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Debug: Print incoming data
        print("Raw Input Data:", data)

        # Convert input to float and binary format where needed
        features = np.array([[
            float(data["age"]), 
            float(data["job"]), 
            float(data["marital"]), 
            float(data["education"]), 
            1 if data["default"].lower() == "yes" else 0, 
            1 if data["housing"].lower() == "yes" else 0, 
            1 if data["loan"].lower() == "yes" else 0
        ]], dtype=float)

        # Debug: Print unscaled input features
        print("Unscaled Features:", features)

        # Ensure feature scaling is applied
        features_scaled = scaler.transform(features)  # type: ignore

        # Debugging: Print processed features
        print("Processed Features (Scaled):", features_scaled)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Debugging: Print raw prediction result
        print("Raw Prediction Output:", prediction)

        # Convert prediction to human-readable result
        result = "yes" if prediction[0] == 1 else "no"

        return jsonify({"subscribed": result})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
