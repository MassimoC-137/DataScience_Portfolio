# 🫀 Heart Disease Prediction - Kaggle Playground S6E2

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Kaggle](https://img.shields.io/badge/Kaggle-Playground-20BEFF)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🇬🇧 Project Overview
This repository contains my solution for the **Kaggle Playground Series - Season 6, Episode 2** competition. The goal is to predict the presence of heart disease using a binary classification approach on a dataset of approx. 600,000 patients.

My approach focuses on robust **Exploratory Data Analysis (EDA)** to identify key risk factors and an **Ensemble Learning** strategy combining Gradient Boosting models.

The project features a **Hardware-Aware Architecture**: the code automatically detects if it's running on a high-end GPU (e.g., Kaggle, NVIDIA Workstation) or a standard CPU (e.g., Mac M-Series, Laptop) and adjusts the model complexity accordingly.

### 🏆 Results
* **Metric:** ROC-AUC
* **CV Score:** 0.9552 (10-Fold Stratified Cross-Validation)
* **Model:** Voting Classifier (Soft Voting) blending **XGBoost** and **LightGBM**.

### 🔑 Key Features & Engineering
Through deep EDA, I engineered several "Golden Features" that significantly boosted model performance:
* **Angina-Thallium Interaction:** Combining exercise-induced angina with thallium stress test results (Highest correlation with target).
* **Hemodynamic Risk:** Interaction between Age and Resting Blood Pressure.
* **MaxHR/Age Ratio:** An indicator of cardiac fitness relative to age.

### 🧠 Methodology
1.  **Exploratory Data Analysis (EDA):** Deep dive into feature distributions and correlations.
2.  **Feature Engineering:** Created "Golden Features" like `Angina_Thallium_Combo` and `Hemodynamic_Risk` based on domain knowledge.
3.  **Adaptive Ensemble Strategy:**
    * **GPU Mode (Kaggle/NVIDIA):** Uses a Voting Classifier with **XGBoost + LightGBM + CatBoost**.
    * **CPU Mode (Mac/Local):** Automatically excludes CatBoost to ensure stability and speed, using an optimized **XGBoost + LightGBM** ensemble.

### 🛠️ Tech Stack
* **Core:** Pandas, NumPy, Scikit-Learn
* **Models:** XGBoost, LightGBM, CatBoost
* **Visualization:** Plotly (Interactive charts)
* **DevOps:** Hardware Detection (`shutil`, `os`), Pipeline Automation.

### 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the notebook `Playground-series-s6e2.ipynb`. The script will automatically detect your hardware and configure the training strategy.

---

## 🇮🇹 Panoramica del Progetto
Questa repository contiene la soluzione per la competizione **Kaggle Playground S6E2**. L'obiettivo è prevedere le malattie cardiache su un dataset di circa 600.000 pazienti.

Il mio approccio si concentra su una **Exploratory Data Analysis (EDA)** approfondita per identificare i fattori di rischio chiave e su una strategia di **Ensemble Learning**.

Il punto di forza del progetto è l'**Architettura Hardware-Aware**: il codice rileva automaticamente se sta girando su una GPU NVIDIA (Kaggle) o su una CPU (Mac M-Series) e adatta l'Ensemble di conseguenza per evitare crash e massimizzare le performance.

### 🏆 Risultati
* **Metrica:** ROC-AUC
* **Punteggio CV:** 0.9552 (Validazione incrociata stratificata a 10 fold)
* **Modello:** Voting Classifier (Soft Voting) che combina **XGBoost** e **LightGBM**.

### 🔑 Feature Engineering Chiave
Attraverso l'analisi dei dati, ho creato diverse nuove variabili ("Golden Features") che hanno migliorato le performance:
* **Interazione Angina-Tallio:** Combina l'angina indotta da sforzo con i risultati del test al tallio (Correlazione più alta col target).
* **Rischio Emodinamico:** Interazione tra Età e Pressione Sanguigna a riposo.
* **Rapporto MaxHR/Età:** Un indicatore della fitness cardiaca relativo all'età del paziente.

### 🧠 Strategia Adattiva
* **Modalità GPU:** Attiva l'Ensemble completo (XGBoost + LightGBM + CatBoost).
* **Modalità CPU (Mac):** Esclude automaticamente CatBoost per garantire stabilità, utilizzando un Ensemble ottimizzato XGBoost + LightGBM.