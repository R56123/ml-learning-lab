# 📊 ML Learning Lab

A hands-on machine learning project built from scratch using Python.

This project predicts student exam scores based on hours studied using linear regression, and includes interactive experiments to explore the effects of key hyperparameters like **learning rate**, **batch size**, and **epochs** on model performance.  
Inspired by Google's Machine Learning Crash Course.

---

## 🔍 Project Overview

- 🎯 **Goal**: Predict student exam scores using a linear regression model
- 🧠 **Learning focus**:
  - Gradient descent behavior
  - Hyperparameter tuning (learning rate, epochs, batch size)
  - Error evaluation (MSE, RMSE)
- 📊 **Dataset**: A small set of sample student scores based on hours studied

---

## 📁 Folder Structure

<pre><code>```txt ml-learning-lab/ ├── data/ │ └── student_scores.csv # Small dataset (hours studied vs score) ├── experiments/ │ ├── experiment_1.py # High learning rate test │ └── experiment_2.py # Low learning rate test ├── model_utils.py # Core ML functions (train, plot, evaluate) ├── main.py # Base model runner ├── requirements.txt # Python dependencies └── README.md # You're here! ```</code></pre>
---

## 🚀 Getting Started

### 1. Clone this repo
```bash
git clone https://github.com/R56123/ml-learning-lab.git
cd ml-learning-lab
pip install -r requirements.txt
python main.py

python experiments/experiment_1.py   # High learning rate
python experiments/experiment_2.py   # Low learning rate


