from flask import Flask, render_template, request
import joblib
import pandas as pd 
# Load the trained model
model = joblib.load("salary_predictor.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data
            age = request.form["age"]
            gender = request.form["gender"]
            education = request.form["education"]
            job = request.form["job"]
            experience = request.form["experience"]

            # Check if any field is empty
            if not all([age, gender, education, job, experience]):
                raise ValueError("All fields are required.")

            # Convert numeric fields
            age = float(age)
            experience = float(experience)

            # Build input DataFrame
            data = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Education Level': education,
                'Job Title': job,
                'Years of Experience': experience
            }])

            prediction = model.predict(data)[0]

        except Exception as e:
            print("Error during prediction:", str(e))
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    app.run(port=8080)
