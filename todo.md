news data from rss feeds is only available for the last 2 years makes the data really small will have to manually scrape the data or find kaggle files for the data 

create api passes for fred and tradingeconomics find how to use newspapers3k or integrate kaggle datasets into this pipeline

## MIDTERM

1) running pipeline for quantised data collection 
2) drop parquet files and fit the data into pgsql timescaleDB
3) NEWS -> kaggle : easy 
        -> rss feeds : easy but tough
        -> manual scraping : very tough but best results
4) prefect etl -> skip 
5) preprocessing -> cleaning -> ffill 
                 -> news data boilerplate 
                 -> html tag removal 
                 -> duplicate removal 
                 -> sceptic view 
                 -> data alignment -> market + news data 
                 -> clustering 
                 -> feature engeeniering 
                 -> data transform 
                 -> exploratory data analysis
                 -> transformers (huggingface)
6) streamlit basic financial dashboard (GUI)
