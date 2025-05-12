# 🧠 ML Learning Lab

A hands-on machine learning lab built from scratch in Python.  
This project showcases simple and advanced regression models using real-world datasets, complete with training visuals and evaluation metrics.

---

## 🚀 Project Highlights

- ✅ Implemented Linear Regression from scratch
- 📊 Visualized training performance using RMSE and MSE over epochs
- 🔍 Compared red vs white wine quality prediction models
- ⏱️ Integrated tqdm progress bars with live terminal updates
- 💾 Interactive chart saving and user input via CLI
- 🎓 Included a student exam score predictor with a single-feature model

---

## 📁 Folder Structure

ml-learning-lab/
├── data/
│ ├── student_scores.csv # Simple dataset (Hours vs Scores)
│ ├── winequality-red.csv # Red wine data
│ └── winequality-white.csv # White wine data
│
├── experiments/
│ ├── experiment_1.py # High learning rate test
│ ├── experiment_2.py # Low learning rate test
│ ├── wine_experiment_red.py # Red wine regression test
│ ├── wine_experiment_white.py # White wine regression test
│ └── compare_wines.py # Side-by-side RMSE/MSE analysis
│
├── model_utils.py # Core ML logic (train, predict, plot)
├── main.py # Run student score model
├── requirements.txt # Python dependencies
└── README.md # You're here!

yaml
Copy
Edit

---

## 🧪 Datasets Used

- **Student Scores** – [simple CSV] `hours studied` → `exam score`
- **Wine Quality (UCI)** – Multifeature dataset:  
  Predicts `quality` based on acidity, sugar, alcohol %, etc.

---

## 📈 Sample Output

Terminal:
Red Wine Epoch 100/100 | MSE: 26.30 | RMSE: 5.13
White Wine Epoch 100/100 | MSE: 29.91 | RMSE: 5.45

yaml
Copy
Edit

Chart:

- RMSE and MSE plotted side-by-side
- Red vs White wine training comparison

---

## 🧠 Learnings

- Difference between MSE (Mean Squared Error) and RMSE (Root MSE)
- Role of hyperparameters (learning rate, epochs)
- Importance of feature scaling (via `StandardScaler`)
- How model loss behaves over training

---

## ▶️ Getting Started

```bash
git clone https://github.com/R56123/ml-learning-lab.git
cd ml-learning-lab
pip install -r requirements.txt
python main.py                         # Run student regression
python experiments/wine_experiment_red.py   # Run wine experiments
python experiments/compare_wines.py         # Compare RMSE & MSE