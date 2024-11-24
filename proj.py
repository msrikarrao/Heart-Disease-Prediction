import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import os

# Global variable to store the model coefficients
coefficients = None
heart_data = pd.DataFrame()  # Empty dataframe to hold the loaded data

# Function to load data from a CSV file
def load_data():
    global heart_data
    file_path = entry_file_path.get()
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        messagebox.showerror("File Error", "The specified file does not exist.")
        return

    try:
        # Load data from CSV
        heart_data = pd.read_csv(file_path)
        
        # all necessary columns are in the data
        required_columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
        if not all(col in heart_data.columns for col in required_columns):
            messagebox.showerror("Data Error", "CSV does not contain the required columns.")
            return

        # Filter out rows with missing values
        heart_data = heart_data.dropna(subset=required_columns)
        
        messagebox.showinfo("Success", "Data loaded successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the data: {e}")

# Function to train the linear regression model using the Normal Equation
def calculate_coefficients_and_predict():
    global coefficients
    if heart_data.empty:
        messagebox.showerror("Data Error", "No data available to train the model.")
        return

    # Prepare the features (X) and target variable (y)
    X = heart_data[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]].values
    y = heart_data["target"].values  # The target is heart disease presence (0 or 1)

    # Add a column of 1s to X for the intercept term (bias)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column for the intercept (bias term)

    try:
        # Compute the coefficients using the Normal Equation
        coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        # Display the coefficients 
        display_coefficients(coefficients)  
        messagebox.showinfo("Model Trained", "Linear regression model has been trained successfully using the Normal Equation!")
    except np.linalg.LinAlgError:
        messagebox.showerror("Math Error", "Matrix inversion failed. The system may be singular (non-invertible).")

# Function to display the coefficients in the GUI (optional)
def display_coefficients(coefficients):
    for i, coef in enumerate(coefficients):
        print(f"Coefficient for feature {i}: {coef:.4f}")

def get_recommendation(risk_score):
    if risk_score < 20:
        return "\nYour risk of heart disease is low. Maintain a healthy lifestyle with a balanced diet and regular exercise."
    elif risk_score < 50:
        return "\nYou have a moderate risk of heart disease. It is advisable to follow a heart-healthy diet, exercise regularly, and monitor your health."
    else:
        return "\nYou have a high risk of heart disease. Please consult with a doctor immediately for further evaluation and potential interventions."

# Function to predict heart disease risk score using the trained model
def predict_disease():
    global coefficients
    if coefficients is None:
        messagebox.showerror("Error", "Model is not trained yet. Please load the data and train the model first.")
        return

    # Get user inputs for prediction
    try:
        name = entry_name.get()  # Get name from the new entry field
        if not name.strip():  # If no name is entered
            messagebox.showerror("Input Error", "Please enter your name.")
            return
        
        age = float(entry_age.get())
        if age < 0 or age > 120:
            messagebox.showerror("Input Error", "Age must be between 0 and 120.")
            return
        
        sex = entry_gender.get().lower()
        if sex not in ['male', 'female']:
            messagebox.showerror("Input Error", "Gender must be 'male' or 'female'.")
            return
        sex = 1 if sex == 'male' else 0
        
        cp = int(entry_cp.get())
        if cp < 0 or cp > 3:
            messagebox.showerror("Input Error", "Chest Pain Type (cp) must be between 0 and 3.")
            return
        
        trestbps = float(entry_trestbps.get())
        if trestbps < 50 or trestbps > 200:
            messagebox.showerror("Input Error", "Resting Blood Pressure (trestbps) must be between 50 and 200.")
            return
        
        chol = float(entry_chol.get())
        if chol < 100 or chol > 600:
            messagebox.showerror("Input Error", "Cholesterol (chol) must be between 100 and 600.")
            return
        
        fbs = entry_fbs.get().lower()
        if fbs not in ['true', 'false']:
            messagebox.showerror("Input Error", "Fasting Blood Sugar (fbs) must be 'true' or 'false'.")
            return
        fbs = 1 if fbs == 'true' else 0
        
        restecg = int(entry_restecg.get())
        if restecg not in [0, 1]:
            messagebox.showerror("Input Error", "Resting Electrocardiographic (restecg) must be 0 or 1.")
            return
        
        thalach = float(entry_thalach.get())
        if thalach < 50 or thalach > 220:
            messagebox.showerror("Input Error", "Max Heart Rate (thalach) must be between 50 and 220.")
            return
        
        exang = entry_exang.get().lower()
        if exang not in ['true', 'false']:
            messagebox.showerror("Input Error", "Exercise Induced Angina (exang) must be 'true' or 'false'.")
            return
        exang = 1 if exang == 'true' else 0
        
        oldpeak = float(entry_oldpeak.get())
        if oldpeak < 0 or oldpeak > 4:
            messagebox.showerror("Input Error", "Old Peak (oldpeak) must be between 0 and 4.")
            return
        
        slope = int(entry_slope.get())
        if slope < 0 or slope > 3:
            messagebox.showerror("Input Error", "Slope must be between 0 and 3.")
            return
        
        ca = int(entry_ca.get())
        if ca not in [0, 1, 2]:
            messagebox.showerror("Input Error", "Number of Major Vessels (ca) must be 0, 1, or 2.")
            return
        
        thal = int(entry_thal.get())
        if thal < 1 or thal > 3:
            messagebox.showerror("Input Error", "Thalassemia (thal) must be between 1 and 3.")
            return
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
        return

    # Prepare the input data in the same format as the model's features
    user_input = np.array([1, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

    # Calculate the prediction (heart disease risk score) using the linear regression model
    risk_score = np.dot(user_input, coefficients)
    risk_score *= 100  # Convert to percentage
    recommendation = get_recommendation(risk_score)
    # Show the prediction result
    messagebox.showinfo("Prediction Result", f"{name}, the predicted risk score of heart disease is: {risk_score:.2f}%.\n {recommendation}")

# Create the main window for the GUI
root = tk.Tk()
root.title("Heart Disease Risk Prediction")

# Create and place labels and entry widgets
tk.Label(root, text="CSV File Path").grid(row=0, column=0, padx=10, pady=5)
entry_file_path = tk.Entry(root, width=50)
entry_file_path.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Name").grid(row=1, column=0, padx=10, pady=5)
entry_name = tk.Entry(root, width=50)
entry_name.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Age").grid(row=2, column=0, padx=10, pady=5)
entry_age = tk.Entry(root, width=50)
entry_age.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Gender (male/female)").grid(row=3, column=0, padx=10, pady=5)
entry_gender = tk.Entry(root, width=50)
entry_gender.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Chest Pain Type (cp) (0-3)").grid(row=4, column=0, padx=10, pady=5)
entry_cp = tk.Entry(root, width=50)
entry_cp.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Resting Blood Pressure (trestbps)").grid(row=5, column=0, padx=10, pady=5)
entry_trestbps = tk.Entry(root, width=50)
entry_trestbps.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Cholesterol (chol)").grid(row=6, column=0, padx=10, pady=5)
entry_chol = tk.Entry(root, width=50)
entry_chol.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Fasting Blood Sugar (fbs) (true/false)").grid(row=7, column=0, padx=10, pady=5)
entry_fbs = tk.Entry(root, width=50)
entry_fbs.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Resting Electrocardiographic (restecg)(0/1)").grid(row=8, column=0, padx=10, pady=5)
entry_restecg = tk.Entry(root, width=50)
entry_restecg.grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="Max Heart Rate (thalach)").grid(row=9, column=0, padx=10, pady=5)
entry_thalach = tk.Entry(root, width=50)
entry_thalach.grid(row=9, column=1, padx=10, pady=5)

tk.Label(root, text="Exercise Induced Angina (exang) (true/false)").grid(row=10, column=0, padx=10, pady=5)
entry_exang = tk.Entry(root, width=50)
entry_exang.grid(row=10, column=1, padx=10, pady=5)

tk.Label(root, text="Old Peak (float 0-4)").grid(row=11, column=0, padx=10, pady=5)
entry_oldpeak = tk.Entry(root, width=50)
entry_oldpeak.grid(row=11, column=1, padx=10, pady=5)

tk.Label(root, text="Slope (0-3)").grid(row=12, column=0, padx=10, pady=5)
entry_slope = tk.Entry(root, width=50)
entry_slope.grid(row=12, column=1, padx=10, pady=5)

tk.Label(root, text="Number of Major Vessels (ca) (0/1)").grid(row=13, column=0, padx=10, pady=5)
entry_ca = tk.Entry(root, width=50)
entry_ca.grid(row=13, column=1, padx=10, pady=5)

tk.Label(root, text="Thalassemia (thal) (1-2)").grid(row=14, column=0, padx=10, pady=5)
entry_thal = tk.Entry(root, width=50)
entry_thal.grid(row=14, column=1, padx=10, pady=5)

# Buttons
tk.Button(root, text="Load Data", command=load_data).grid(row=15, column=0, columnspan=2, padx=10, pady=5)
tk.Button(root, text="Train Model", command=calculate_coefficients_and_predict).grid(row=16, column=0, columnspan=2, padx=10, pady=5)
tk.Button(root, text="Predict Risk", command=predict_disease).grid(row=17, column=0, columnspan=2, padx=10, pady=5)

# Start the main loop
root.mainloop()
