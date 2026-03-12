# FinLagX Project Plan

**Last Updated:** March 8, 2026

---

## 🎯 Current Phase: Advanced Modeling & Analysis

The project has successfully completed the data pipelines, deep learning baseline, and the interactive Streamlit dashboard. The current focus is on building out Tree-Based models and exploring feature importance.

---

## ✅ Completed Milestones

- [x] **Setup & Infrastructure**: Docker environment with TimescaleDB and MongoDB is fully operational.
- [x] **Data Ingestion Pipeline**: Automated scripts for market, macro, and news data collection are in place.
- [x] **Data Preprocessing**: Initial data cleaning and alignment pipelines are functional.
- [x] **Statistical Modeling**: Implemented `statsmodels` for Granger Causality and VAR analysis.
- [x] **Deep Learning Modeling**: Implemented `PyTorch` LSTM for time-series forecasting.
- [x] **Experiment Tracking**: Integrated `MLflow` for tracking and visualizing model performance.
- [x] **Version Control**: Modeling work has been pushed to the `modelingdev` branch on GitHub.
- [x] **Sentiment Analysis**: Implemented `FinBERT` pipeline to generate and store sentiment scores for news articles.
- [x] **Dashboarding**: Streamlit dashboard successfully built and integrated with model outputs.
- [x] **Baseline Tree Models**: Created Tabular LightGBM model utilizing rolling windows to beat deep learning baseline.

---

## ⏳ Next Steps & To-Do

### 1. ~~**Build Final Feature Dataset (Evening Task)**~~ ✅ COMPLETED (March 3, 2026)

    - **Priority**: ~~High~~ → Done
    - **Task**: Re-implemented `src/preprocessing/build_features.py`.
    - **Result**: Pipeline produces a unified dataset ready for the modeling phase.

### 2. ~~**Re-run Models with Sentiment Data**~~ ✅ COMPLETED (March 3, 2026)

    - **Priority**: ~~High~~ → Done
    - **Task**: After the feature dataset is built, re-run the `market_modeling.py` and `pytorch_modeling.py` scripts.
    - **Goal**: Analyze the impact of the new sentiment feature on model performance and generate updated results.

### 3. ~~**Develop the Streamlit Dashboard**~~ ✅ COMPLETED (March 8, 2026)

    - **Priority**: ~~Medium~~ → Done
    - **Task**: Begin development of the `Streamlit` dashboard to visualize project outputs.
    - **Initial Features**:
        - ✅ Display the lead-lag dependency graph from the `statsmodels` analysis.
        - ✅ Show the `PyTorch` model's predictions vs. actuals chart.
        - ✅ Create interactive charts for exploring the raw market and macro data, including news sentiment.

---

## 🗃️ Backlog & Future Enhancements

### 1. Machine Learning Model Improvements

- ~~**Introduce Tree-Based Benchmarks (LightGBM/XGBoost)**~~: ✅ COMPLETED (March 8, 2026). Implemented Gradient Boosting trees as a strong baseline model, framing time-series forecasting as a tabular learning problem.
- **Time-Series Transformers**: Implement a Transformer-based architecture specifically designed for time-series forecasting.
- **Target Engineering (Direction First, Magnitude Second)**: ✅ COMPLETED (March 12, 2026). Implemented Two-Stage Hurdle Model (LightGBM Classifier + Regressor).

### 2. Dataset & Feature Engineering Improvements

- **Solve News Data Limitation**: Obtain historical news data (2+ years) to better evaluate sentiment impact.
- **Refine Preprocessing & Advanced Indicators**:
  - ~~Implement advanced technical indicators (Momentum, RSI, MACD, Bollinger Bands, ATR) for better short-term signaling.~~ ✅ COMPLETED (March 11, 2026)
  - Explore clustering techniques to group similar assets. ✅ COMPLETED (March 12, 2026). Implemented K-Means clustering in `src/preprocessing/asset_clustering.py`.

### 3. MLOps & Orchestration

- **Hyperparameter Tuning**: ✅ COMPLETED (March 12, 2026). Integrated Optuna for automated search and logging to MLflow.
- **Orchestration**: Re-evaluate integrating **Prefect** for more robust ETL pipeline orchestration.
