# 🩺 Disease Prediction Web App

This is a simple AI-powered web application built using **Streamlit** that predicts whether a person has **Diabetes** or **Heart Disease** based on user input. It uses **Random Forest classifiers** trained on real medical datasets.

---

## 🚀 Features

- 🧠 Predicts **Diabetes** and **Heart Disease**
- 📊 Interactive UI with form-based input
- 💡 Categorical feature encoding handled internally
- 🧪 Trained using Scikit-learn Random Forest
- 🖥️ Built with Streamlit for rapid UI deployment

---

## 🗂️ Folder Structure

disease-prediction-app/
├── app.py
├── train_diabetes_model.py
├── train_heart_model.py
├── models/
│ ├── diabetes_model.pkl
│ └── heart_model.pkl
├── data/
│ ├── diabetes.csv
│ └── heart.csv
└── README.md


> 🔁 **Note**: `train_*.py` scripts expect the `.csv` files in the `data/` folder. The Streamlit `app.py` loads `.pkl` models from the `models/` folder.

---

## 📦 Installation

### 🛠 Requirements
- Python 3.7+
- Pandas
- Numpy
- Scikit-learn
- Streamlit

### ⚙️ Setup Instructions

```bash
git clone https://github.com/your-username/disease-prediction-app.git
cd disease-prediction-app

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Train models (optional if .pkl files already exist)
python train_diabetes_model.py
python train_heart_model.py

# Run the app
streamlit run app.py
