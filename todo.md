    # FinLagX Project Plan

    **Last Updated:** October 21, 2025

    ---

    ## 🎯 Current Phase: Model Integration & Visualization

    The project has successfully completed all data collection and feature engineering phases, including sentiment analysis. The immediate focus is on integrating this new sentiment feature into the models and presenting the final results in a user-friendly dashboard.

    ---

    ##   Completed Milestones

    -   [x] **Setup & Infrastructure**: Docker environment with TimescaleDB and MongoDB is fully operational.
    -   [x] **Data Ingestion Pipeline**: Automated scripts for market, macro, and news data collection are in place.
    -   [x] **Data Preprocessing**: Initial data cleaning and alignment pipelines are functional.
    -   [x] **Statistical Modeling**: Implemented `statsmodels` for Granger Causality and VAR analysis.
    -   [x] **Deep Learning Modeling**: Implemented `PyTorch` LSTM for time-series forecasting.
    -   [x] **Experiment Tracking**: Integrated `MLflow` for tracking and visualizing model performance.
    -   [x] **Version Control**: Modeling work has been pushed to the `modelingdev` branch on GitHub.
    -   [x] **Sentiment Analysis**: Implemented `FinBERT` pipeline to generate and store sentiment scores for news articles.

    ---

    ## ⏳ Next Steps & To-Do

    ### 1. ~~**Build Final Feature Dataset (Evening Task)**~~ ✅ COMPLETED (March 3, 2026)
        - **Priority**: ~~High~~ → Done
        - **Task**: Re-implemented `src/preprocessing/build_features.py`:
            - ✅ Fetches market data from TimescaleDB (via MarketDataPreprocessor).
            - ✅ Fetches news articles and their FinBERT sentiment scores from MongoDB.
            - ✅ Aggregates daily sentiment (mean, std, min, max, count per category).
            - ✅ Aligns and merges market + sentiment on a daily basis (left join).
            - ✅ Adds derived features (SMA crossover, volatility regime, return sign).
            - ✅ Saves the final feature-rich dataset to `data/processed/market/aligned_market_data.parquet`.
        - **Result**: Pipeline produces a unified dataset ready for the modeling phase.

    ### 2. ~~**Re-run Models with Sentiment Data**~~ ✅ COMPLETED (March 3, 2026)
        - **Priority**: ~~High~~ → Done
        - **Task**: After the feature dataset is built, re-run the `market_modeling.py` and `pytorch_modeling.py` scripts.
        - **Goal**: Analyze the impact of the new sentiment feature on model performance and generate updated results (graphs, metrics in MLflow).

    ### 3. **Develop the Streamlit Dashboard**
        - **Priority**: Medium
        - **Task**: Begin development of the `Streamlit` dashboard to visualize project outputs.
        - **Initial Features**:
            - Display the lead-lag dependency graph from the `statsmodels` analysis.
            - Show the `PyTorch` model's predictions vs. actuals chart.
            - Create interactive charts for exploring the raw market and macro data, including news sentiment.

    ---

    ##  backlog & Future Enhancements

    -   **Solve News Data Limitation**: The RSS feeds provide limited historical data (~2 years).
        - **Option A**: Integrate a large, pre-existing news dataset from **Kaggle**.
        - **Option B**: Develop a more robust web scraper using `BeautifulSoup` to manually collect historical news archives.
    -   **Refine Preprocessing**:
        - Implement more advanced feature engineering (e.g., volatility metrics, technical indicators).
        - Explore clustering techniques to group similar assets.
    -   **Enhance Modeling**:
        - Experiment with more advanced `PyTorch` models like **Transformers**.
        - Tune hyperparameters for the LSTM model using MLflow's tracking capabilities.
    -   **Orchestration**: Re-evaluate integrating **Prefect** for more robust ETL pipeline orchestration.
