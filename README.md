# 🇵🇱 Bankruptcy Prediction in Polish Companies: A Machine Learning Approach

This project explores the use of **Machine Learning (ML)** models to predict corporate bankruptcy in Polish companies using real-world financial data[cite: 10]. The goal is to identify early warning signs of financial distress, enabling banks, investors, and companies to make proactive, informed decisions[cite: 9, 272].

---

## 🎯 Project Goals

* Predict whether a company will go bankrupt or not using machine learning models trained on real-world financial indicators[cite: 10].
* Address the challenge of a highly imbalanced dataset[cite: 29].
* Determine the best-performing classification model among Logistic Regression, Random Forest, XGBoost, and CatBoost[cite: 15, 11, 12].
* Use SHAP (SHapley Additive exPlanations) analysis on the final model to uncover how individual financial features influence predictions, providing transparency and actionable conclusions[cite: 13, 16, 17].

---

## 💡 Key Findings & Methodology

### 📊 Dataset & Preprocessing

* **Source:** UCI Machine Learning Repository.
**Focus:** Bankruptcy prediction of Polish companies analyzed during the period 2000-2013[cite: 20, 27].
* **Data Size:** 43,398 company records and 66 financial attributes[cite: 28].
* **Class Imbalance:** Highly imbalanced, with only **4.82%** of companies classified as bankrupt (and 95.18% as non-bankrupt)[cite: 29].
* **Handling Imbalance:** **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training dataset, resulting in a balanced dataset with 35,132 samples for each class[cite: 112, 114].
* **Other Steps:** Median imputation was applied for missing values, and `StandardScaler` was used to standardize features[cite: 108, 111].

### 🧠 Model Performance Summary

The models were evaluated, and the **XGBoost** model demonstrated the strongest performance in distinguishing between classes:

| Model | Accuracy | Precision (Bankrupt Class) | Recall (Bankrupt Class) | ROC AUC Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 70.66% [cite: 120] | 11% [cite: 129] | 64% [cite: 129] | 0.74 [cite: 142] |
| **Random Forest** | 92.8% [cite: 172] | 34% [cite: 178] | 32% [cite: 178] | 0.85 [cite: 190] |
| **XGBoost** (Final Model) | $\approx$ 96% [cite: 240] | **63%** [cite: 244] | 53% [cite: 245] | **0.94** [cite: 238] |
| **CatBoost** | $\approx$ 94% [cite: 263] | 44% [cite: 265] | **59%** [cite: 267] | $\approx$ 0.92 [cite: 263] |

**Conclusion:** **XGBoost** was selected as the final predictive model[cite: 268]. Its AUC of 0.94 indicates a solid discriminative power[cite: 238].

### 🔎 SHAP Analysis (Top Financial Indicators)

SHAP analysis was performed on the XGBoost model to provide transparency and explainability[cite: 275]. The results align with traditional financial and business logic[cite: 275, 269].

| Rank | Feature | Description | Business Logic / Significance |
| :--- | :--- | :--- | :--- |
| **1** | **sales / receivables** | Measures efficiency in collecting on credit sales[cite: 276, 277]. | Cash flow problems arise when struggling to collect from customers; an early sign of distress[cite: 278]. |
| **2** | **(receivables \* 365) / sales** | Day sales outstanding[cite: 279]. | The model treats this ratio (and its inverse) with high importance[cite: 280]. |
| **3** | **(current assets - inventory) / short-term liabilities** | The Quick Ratio (or Acid-Test Ratio)[cite: 281]. | Measures a company's ability to pay short-term debts without relying on inventory sales[cite: 281]. |
| **4** | **retained earnings / total assets** | Measures assets financed by retained profits[cite: 286]. | This is a key component (**B**) of the classic **Altman Z-Score** for calculating credit risk[cite: 286, 285]. |

---

## 💻 Repository Structure

* `data/`: Raw and intermediate datasets.
* `notebooks/`: Jupyter notebooks detailing the data exploration, preprocessing, model training, and SHAP analysis.
* `reports/`: The full project report (`Report_Bankruptcy_prediction_in_poland.pdf`).
* `src/`: Any reusable Python scripts or modules.

---

## ⚙️ Dependencies

This project relies on standard data science libraries, including:

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `catboost`
* `shap`
