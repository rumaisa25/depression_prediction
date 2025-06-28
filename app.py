from flask import Flask, render_template, request
import numpy as np
import pickle

# Create Flask app
app = Flask(__name__)

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        gender = int(request.form["gender"])
        age = int(request.form["age"])
        cgpa = float(request.form["cgpa"])
        residential_status = int(request.form["residential_status"])
        average_sleep = float(request.form["average_sleep"])
        academic_workload = int(request.form["academic_workload"])
        financial_concerns = int(request.form["financial_concerns"])
        social_relationships = int(request.form["social_relationships"])
        anxiety = int(request.form["anxiety"])
        future_insecurity = int(request.form["future_insecurity"])

        # Combine into one feature vector
        features = np.array([[gender, age, cgpa, residential_status, average_sleep,
                              academic_workload, financial_concerns, social_relationships,
                              anxiety, future_insecurity]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert result to message
        result = "⚠️ Likely Depressed" if prediction == 1 else "😊 Not Depressed"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
