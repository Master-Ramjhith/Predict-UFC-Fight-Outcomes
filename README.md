# 🥊 UFC Fight Outcome Prediction (1994–2025)

### UE23CS352A – Machine Learning Mini Project  
**Problem Number:** 74  
**Team Members:**  
- **V C RAMJHITH – PES1UG23CS662**  
- **Tanya Tripathi – PES1UG23CS638**  
**Institution:** PES University, RR Campus, Bengaluru, Karnataka, India  

---

## 🧠 Project Overview

This project predicts the outcome of **UFC fights** (Red vs Blue corner) using supervised **machine learning algorithms** trained on UFC fight statistics from **1994 to 2025**.  
Five models were trained and compared to identify the most accurate predictor.

**Models Used:**  
- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

---

## ⚙️ How to Run the Project

You can run this project in **Google Colab** or **Jupyter Notebook** using the provided file:
```

UFC_Fight_Outcome_Prediction.ipynb

````

### 🧩 Step-by-Step Execution

#### 1️⃣ Open in Google Colab
- Upload the notebook to [Google Colab](https://colab.research.google.com/).  
- Ensure runtime type is set to **Python 3**.

#### 2️⃣ Download the Dataset
Dataset Source: [UFC Datasets 1994–2025 on Kaggle](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)

You can load it in two ways:

##### 🔹 Option A – Manual Upload
Upload the dataset manually in Colab:
```python
from google.colab import files
uploaded = files.upload()  # Select ufc_dataset.csv
````

##### 🔹 Option B – Kaggle Import (Recommended)

Use your Kaggle API token (`kaggle.json`) to download the dataset directly:

```bash
!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d neelagiriaditya/ufc-datasets-1994-2025
!unzip ufc-datasets-1994-2025.zip
```

Ensure the file name is **`ufc_dataset.csv`** and it’s in the notebook’s working directory.

#### 3️⃣ Install Dependencies

If running locally (e.g., Jupyter), install all dependencies:

```bash
pip install -r requirements.txt
```

#### 4️⃣ Run All Cells

Execute all cells in order.
The notebook will:

* Load & preprocess data
* Train five models
* Compare metrics (Accuracy, F1 Score, AUC)
* Display confusion matrix, ROC curve, and feature importances

---

## 📊 Results Summary

| Model               | Accuracy  | F1 Score  | AUC       | Time (s) |
| ------------------- | --------- | --------- | --------- | -------- |
| **Random Forest**   | **0.601** | **0.702** | **0.584** | 3.45     |
| XGBoost             | 0.574     | 0.675     | 0.581     | 6.15     |
| CatBoost            | 0.611     | 0.713     | 0.572     | 43.20    |
| Logistic Regression | 0.507     | 0.560     | 0.519     | 4.63     |
| LightGBM            | 0.541     | 0.649     | 0.509     | 1.08     |

🏆 **Best Model:** Random Forest Classifier

* Accuracy ≈ 60 %
* F1 Score ≈ 0.70
* AUC ≈ 0.58

---

## 🧮 Data Summary

| Property           | Value                      |
| ------------------ | -------------------------- |
| Records            | 1,477 fights               |
| Raw Columns        | 895                        |
| Processed Features | 1,310                      |
| Target             | Winner (1 = Red, 0 = Blue) |

---

## 📂 Repository Structure

```
UFC-Fight-Outcome-Prediction/
│
├── UFC_Fight_Outcome_Prediction.ipynb   # Main notebook
├── ufc_prediction.py                    # Optional Python script version
├── requirements.txt                     # Dependencies list
├── README.md                            # Project documentation
└── ufc_dataset.csv                      # (Upload or download separately)
```

---

## 📈 Generated Outputs

* ✅ Model comparison summary
* ✅ Confusion matrix heatmap
* ✅ ROC curve visualization
* ✅ Top 15 feature importances
* ✅ Final summary report with metrics

---

## ⚠️ Challenges

* Slight class imbalance toward Red-corner wins (~59 %).
* High dimensionality (~1,300 features) increased model complexity.
* Missing external factors (strategy, injuries, venue) limited predictive power.

---

## 🚀 Future Enhancements

* Apply **stacked ensemble learning** for better generalization.
* Model **fighter performance trends** using LSTM/transformer architectures.
* Integrate **external contextual data** (venue, referee, injury history).
* Use **feature selection** (PCA, mutual information) to reduce complexity.

---

## 📚 References

* [UFC Datasets 1994–2025 on Kaggle](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
* [scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/) | [LightGBM](https://lightgbm.readthedocs.io/) | [CatBoost](https://catboost.ai/)

---

## 👥 Authors

| Name               | Roll No       | Contribution                                   |
| ------------------ | ------------- | ---------------------------------------------- |
| **V C RAMJHITH**   | PES1UG23CS662 | Data Preprocessing & Model Implementation      |
| **Tanya Tripathi** | PES1UG23CS638 | Feature Engineering & Evaluation Visualization |

---

## 🏁 Conclusion

The **Random Forest Classifier** achieved the best overall balance of accuracy, interpretability, and runtime.
Although moderate in predictive power, this project demonstrates a complete ML workflow for combat-sports analytics.

> *“Data may not decide who wins a fight — but it shows who’s more likely to.”*

````

---

## 📦 `requirements.txt`

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
````
