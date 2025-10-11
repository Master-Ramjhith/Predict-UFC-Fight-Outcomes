# 🥊 UFC Fight Outcome Prediction (1994–2025)

### UE23CS352A – Machine Learning Mini-Project  
**Problem Number:** 74  
**Team Members:**  
- **V C RAMJHITH** – PES1UG23CS662  
- **Tanya Tripathi** – PES1UG23CS638  
**Institution:** PES University, RR Campus, Bengaluru (Karnataka, India)

---

## 🧠 Project Overview
This project predicts the winner of a UFC fight (**Red corner vs Blue corner**) using **supervised machine-learning models** trained on fighter statistics from 1994 to 2025.  
Five algorithms were benchmarked to identify the most reliable predictor of fight outcomes.

### ✅ Models Compared
1. **Logistic Regression**  
2. **Random Forest Classifier**  
3. **XGBoost Classifier**  
4. **LightGBM Classifier**  
5. **CatBoost Classifier**

---

## ⚙️ Environment Setup (Google Colab)

You can run this entire workflow in **Google Colab** using one of two dataset-upload options.

### 🔹 Option 1 — Manual CSV Upload
1. Open the notebook in Google Colab.  
2. Upload your dataset file manually when prompted:
   ```python
   from google.colab import files
   uploaded = files.upload()  # upload ufc_dataset.csv
````

3. Ensure the file name matches exactly:

   ```
   ufc_dataset.csv
   ```

### 🔹 Option 2 — Kaggle Import (Recommended)

1. Upload your `kaggle.json` API token to Colab:

   ```python
   from google.colab import files
   files.upload()  # select kaggle.json
   ```
2. Configure and download the dataset:

   ```bash
   !mkdir ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !pip install -q kaggle
   !kaggle datasets download -d neelagiriaditya/ufc-datasets-1994-2025
   !unzip ufc-datasets-1994-2025.zip
   ```
3. Confirm that the extracted CSV is named `ufc_dataset.csv`.

---

## 🧩 How to Run the Project

1. **Install dependencies** (Colab preinstalls most, but you can ensure all are present):

   ```bash
   pip install -r requirements.txt
   ```
2. **Upload or import the dataset** (choose one of the two options above).
3. **Run the script or notebook:**

   ```python
   # If running as a script
   !python ufc_prediction.py
   ```

   or execute each cell sequentially in Colab.
4. The notebook will:

   * Load and preprocess the data
   * Train all five models
   * Compare their performance
   * Display metrics, confusion matrix, ROC curve, and feature importance

---

## 🧮 Data Summary

| Property               | Value                      |
| ---------------------- | -------------------------- |
| **Records**            | 1,477 fights               |
| **Raw Columns**        | 895                        |
| **Processed Features** | 1,310                      |
| **Target**             | Winner (1 = Red, 0 = Blue) |

---

## 🔍 Methodology

### 1️⃣ Preprocessing

* **Imputation:** Median for numeric, constant for categorical
* **Scaling:** StandardScaler
* **Encoding:** OneHotEncoder for categorical features
* **Feature Engineering:** Created relative-difference features (e.g., `height_diff`, `reach_diff`) and limited interaction terms
* **Dropped:** Identifiers and irrelevant columns (name, id, location, etc.)

### 2️⃣ Modeling

* **Train/Test Split:** 80 / 20 (stratified by winner)
* **Evaluation Metrics:** Accuracy | F1 Score | ROC-AUC
* **Class Weights:** Balanced to handle slight Red-corner bias

### 3️⃣ Results Summary

| Model               | Accuracy  | F1 Score  | AUC       | Time (s) |
| ------------------- | --------- | --------- | --------- | -------- |
| **Random Forest**   | **0.601** | **0.702** | **0.584** | 3.45     |
| XGBoost             | 0.574     | 0.675     | 0.581     | 6.15     |
| CatBoost            | 0.611     | 0.713     | 0.572     | 43.2     |
| Logistic Regression | 0.507     | 0.560     | 0.519     | 4.63     |
| LightGBM            | 0.541     | 0.649     | 0.509     | 1.08     |

🏆 **Best Model:** Random Forest Classifier

* Accuracy ≈ 60.1 %
* F1 Score ≈ 0.702
* AUC ≈ 0.584
* Runtime < 4 s

---

## 📊 Generated Outputs

* **Model Comparison Table** – shows metrics for all models
* **Confusion Matrix** – visual comparison of predictions vs actual outcomes
* **ROC Curve** – plots true positive rate vs false positive rate
* **Feature Importance Plot** – top 15 attributes impacting fight outcome
* **Summary Report** – aggregated model statistics and runtime

---

## ⚠️ Challenges

* **Imbalanced Classes:** ~59 % Red-corner wins ⇒ used `class_weight='balanced'`
* **High Dimensionality:** 1,310 features increased training time and risk of overfitting
* **External Factors Missing:** Strategy, injuries, and psychological variables not captured in data

---

## 🚀 Future Enhancements

* Introduce **stacked ensembles** for higher AUC
* Model **time-series fighter histories** (LSTM or Transformer approach)
* Include **external contextual features** (weather, venue, injuries)
* Apply **feature selection and dimensionality reduction** (PCA, mutual information)

---

## 📚 References

* Kaggle Dataset: [UFC Datasets 1994–2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
* scikit-learn Docs → [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
* CatBoost Docs → [https://catboost.ai/](https://catboost.ai/)
* LightGBM Docs → [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)
* XGBoost Docs → [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

---

## 👥 Authors

| Name               | Roll No       | Contribution                                   |
| ------------------ | ------------- | ---------------------------------------------- |
| **V C RAMJHITH**   | PES1UG23CS662 | Data Preprocessing & Model Implementation      |
| **Tanya Tripathi** | PES1UG23CS638 | Feature Engineering & Evaluation Visualization |

---

## 🏁 Conclusion

The **Random Forest Classifier** achieved the best trade-off between accuracy and interpretability, proving effective for UFC fight outcome prediction.
Although the predictive power is moderate, the framework is a strong foundation for integrating richer contextual data in future iterations.

---

> *“Data may not decide who wins a fight—but it tells who is more likely to.”*
