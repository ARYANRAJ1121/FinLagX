from src.data.market_data import download_all_assets
from src.data.macro_data import download_all_macro
from src.data.news_data import download_all_news
#from src.data.preprocessing.market_preprocessing import process_and_analyze_market_data
#from data.macro_api import run_macro_api
#from data.tradingeconomics_api import fetch_te_news 
if __name__ == "__main__":
    print("🚀 Starting FinLagX Data Pipeline...\n")
    
    download_all_assets()
    #download_all_macro()
    #download_all_news()
    #run_macro_api()
    #fetch_te_news()
    #process_and_analyze_market_data()
    
    
    print("\n✅ Data pipeline finished.")
