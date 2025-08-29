🩺 Disease Prediction Toolkit

📌 Overview

The Disease Prediction Toolkit is a machine learning project designed to predict the likelihood of diseases based on patient health data.
This implementation focuses on Heart Disease Prediction using a combination of data preprocessing, feature engineering, and classification models.

The project is built in Python with Jupyter Notebooks for experimentation and leverages popular ML libraries for training, evaluation, and deployment.

🚀 Features

Data preprocessing and cleaning for healthcare datasets

Handling categorical, numerical, and boolean features

Training ML models (Logistic Regression, Random Forest, etc.)

Model evaluation with accuracy, confusion matrix, and metrics

Saving & reusing trained models with joblib

Simple interface via Jupyter Notebook (and optional Streamlit app)

🛠️ Technologies Used

Python 3.x

NumPy, Pandas → Data handling

Scikit-learn → Machine learning algorithms

Matplotlib, Seaborn → Data visualization

Joblib → Model persistence (save/load)

(Optional) Streamlit → Interactive web app
📂 Project Structure
Disease_Prediction_Toolkit/
│── notebooks/
│   ├── 01_data_exploration.ipynb   # Exploratory Data Analysis
│   ├── 02_model_training.ipynb     # Model training & evaluation
│   └── 03_prediction_demo.ipynb    # Prediction workflow
│
│── models/                         # Saved trained models & scalers
│   ├── heart_rf_model.pkl
│   ├── heart_scaler.pkl
│   └── X_columns.pkl
│
│── data/
│   └── Heart_dataset.csv           # Dataset (if available)
│
│── src/
│   ├── preprocess.py               # Preprocessing functions
│   ├── train_model.py              # Training script
│   └── predict.py                  # Prediction script
│
│── app.py                          # Streamlit app (optional)
│── requirements.txt                # Dependencies
│── README.md                       # Project documentation
⚙️ Installation

1. Clone the repository: git clone https://github.com/your-username/Disease_Prediction_Toolkit.git cd Disease_Prediction_Toolkit
2. Create and activate a virtual environment (recommended): python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
3. Install dependencies:pip install -r requirements.txt
▶️ Usage
Run in Jupyter Notebook

Open notebooks/02_model_training.ipynb

Train the model & evaluate results

Use the saved model for predictions

Run the Prediction Script
python src/predict.py --input data/new_patient.csv

Launch the Streamlit Web App (optional)
streamlit run app.py

📊 Example Workflow

1. Load dataset

2. Preprocess data (handle missing values, encode categories, scale features)
import pandas as pd

df = pd.read_csv("data/Heart_dataset.csv")

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop("target", axis=1))
y = df_encoded["target"]

3. Train ML model (Random Forest / Logistic Regression)

4. Evaluate results (accuracy, confusion matrix, classification report)

5. Save trained model & scaler for deployment

6. Use model for predictions on new patient data

🔮 Future Improvements

Expand dataset for multiple disease categories (diabetes, cancer, etc.)

Build a full-featured Streamlit web app

Add deep learning models (Neural Networks)

Deploy toolkit on Cloud (AWS, GCP, or HuggingFace Spaces)

👨‍💻 Author

Abhirami R K
📧 Email – ammuram576@gmail.com
