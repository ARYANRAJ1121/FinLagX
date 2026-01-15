"""
News Data Preprocessing - MongoDB-aware
Cleans and prepares news articles for sentiment analysis
"""
import pandas as pd
import re
from pymongo import MongoClient
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_CONFIG = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', '27017')),
    'username': os.getenv('MONGO_USER', 'admin'),
    'password': os.getenv('MONGO_PASSWORD', 'finlagx_mongo'),
    'database': os.getenv('MONGO_DB', 'finlagx_news')
}

class NewsDataPreprocessor:
    """Preprocesses news data from MongoDB"""
    
    def __init__(self):
        self.client = self._get_mongo_client()
        self.db = self.client[MONGO_CONFIG['database']]
        self.collection = self.db.news_articles
    
    def _get_mongo_client(self):
        """Get MongoDB client connection"""
        try:
            client = MongoClient(
                host=MONGO_CONFIG['host'],
                port=MONGO_CONFIG['port'],
                username=MONGO_CONFIG['username'],
                password=MONGO_CONFIG['password'],
                authSource='admin'
            )
            client.admin.command('ismaster')
            return client
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def get_news_data(self, categories=None, start_date=None, end_date=None):
        """
        Fetch news data from MongoDB
        
        Args:
            categories: List of categories to fetch
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with news articles
        """
        query = {}
        
        if categories:
            query["source.category"] = {"$in": categories}
        
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        logger.info(f"Fetching news data from MongoDB...")
        cursor = self.collection.find(query).sort("timestamp", 1)
        
        articles = list(cursor)
        df = pd.DataFrame(articles)
        
        if not df.empty:
            logger.info(f"  Fetched {len(df)} articles")
        else:
            logger.warning("  No news data found")
        
        return df
    
    def clean_text(self, text):
        """
        Clean text data: remove HTML, special characters, extra spaces
        
        Args:
            text: Raw text string
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_data(self, df):
        """
        Clean news data: remove duplicates, clean text
        
        Args:
            df: Raw news DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("🧹 Cleaning news data...")
        
        original_rows = len(df)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicates based on URL
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'], keep='first')
            logger.info(f"   Removed {original_rows - len(df)} duplicate URLs")
        
        # Clean title and summary
        if 'title' in df.columns:
            df['title_clean'] = df['title'].apply(self.clean_text)
        
        if 'summary' in df.columns:
            df['summary_clean'] = df['summary'].apply(self.clean_text)
        
        # Remove articles with empty cleaned text
        if 'title_clean' in df.columns and 'summary_clean' in df.columns:
            df = df[
                (df['title_clean'].str.len() > 0) | 
                (df['summary_clean'].str.len() > 0)
            ]
        
        # Extract category from nested source field
        if 'source' in df.columns:
            df['category'] = df['source'].apply(
                lambda x: x.get('category', 'unknown') if isinstance(x, dict) else 'unknown'
            )
            df['source_name'] = df['source'].apply(
                lambda x: x.get('name', 'unknown') if isinstance(x, dict) else 'unknown'
            )
        
        cleaned_rows = len(df)
        logger.info(f"  Cleaned: {original_rows} → {cleaned_rows} articles")
        
        return df
    
    def extract_keywords(self, df, top_n=10):
        """
        Extract keywords from articles (simple frequency-based)
        
        Args:
            df: Cleaned news DataFrame
            top_n: Number of top keywords per article
        
        Returns:
            DataFrame with keywords extracted
        """
        logger.info("🔑 Extracting keywords...")
        
        from collections import Counter
        import nltk
        
        # Download stopwords if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add financial stopwords
        financial_stopwords = {'said', 'says', 'will', 'could', 'would', 'may', 'might'}
        stop_words.update(financial_stopwords)
        
        def get_keywords(text, n=top_n):
            if not isinstance(text, str) or len(text) == 0:
                return []
            
            # Tokenize and filter
            words = text.lower().split()
            words = [w for w in words if w not in stop_words and len(w) > 3]
            
            # Get top N most common
            counter = Counter(words)
            return [word for word, _ in counter.most_common(n)]
        
        # Combine title and summary for keyword extraction
        df['combined_text'] = df['title_clean'].fillna('') + ' ' + df['summary_clean'].fillna('')
        df['keywords'] = df['combined_text'].apply(get_keywords)
        
        logger.info("  Keywords extracted")
        
        return df
    
    def prepare_for_sentiment(self, df):
        """
        Prepare articles for sentiment analysis
        
        Args:
            df: Cleaned news DataFrame with keywords
        
        Returns:
            DataFrame ready for sentiment analysis
        """
        logger.info("📝 Preparing for sentiment analysis...")
        
        # Select relevant columns
        sentiment_df = df[[
            'article_id', 'timestamp', 'category', 'source_name',
            'title', 'title_clean', 'summary_clean', 'url', 'keywords'
        ]].copy()
        
        # Create combined text for sentiment (use cleaned version)
        sentiment_df['text_for_sentiment'] = (
            sentiment_df['title_clean'].fillna('') + ' ' + 
            sentiment_df['summary_clean'].fillna('')
        )
        
        # Filter out very short articles (less than 20 characters)
        sentiment_df = sentiment_df[sentiment_df['text_for_sentiment'].str.len() >= 20]
        
        logger.info(f"  Prepared {len(sentiment_df)} articles for sentiment analysis")
        
        return sentiment_df
    
    def save_to_mongodb(self, df, collection_name='news_articles_processed'):
        """
        Save processed news back to MongoDB
        
        Args:
            df: Processed DataFrame
            collection_name: Target collection name
        """
        logger.info(f"💾 Saving processed news to {collection_name}...")
        
        try:
            # Drop existing collection
            if collection_name in self.db.list_collection_names():
                self.db[collection_name].drop()
            
            # Convert DataFrame to dict records
            records = df.to_dict('records')
            
            # Insert into MongoDB
            if records:
                self.db[collection_name].insert_many(records)
                logger.info(f"  Saved {len(records)} articles to {collection_name}")
            else:
                logger.warning("  No records to save")
                
        except Exception as e:
            logger.error(f"  Error saving to MongoDB: {e}")
            raise
    
    def run_full_preprocessing(self, categories=None, start_date=None, end_date=None, 
                               save=True, collection_name='news_articles_processed'):
        """
        Run the complete news preprocessing pipeline
        
        Args:
            categories: List of categories to process
            start_date: Start date
            end_date: End date
            save: Whether to save results
            collection_name: MongoDB collection name for saving
        
        Returns:
            Processed DataFrame
        """
        logger.info("  Starting news preprocessing pipeline...\n")
        
        # 1. Fetch data
        df = self.get_news_data(categories, start_date, end_date)
        
        if df.empty:
            logger.warning("  No news data to process")
            return df
        
        # 2. Clean data
        df = self.clean_data(df)
        
        # 3. Extract keywords
        df = self.extract_keywords(df)
        
        # 4. Prepare for sentiment
        df = self.prepare_for_sentiment(df)
        
        # 5. Save to MongoDB
        if save:
            self.save_to_mongodb(df, collection_name)
        
        logger.info("\n  News preprocessing completed!")
        logger.info(f"   Final articles: {len(df)}")
        logger.info(f"   Categories: {df['category'].unique().tolist()}")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df

def get_processed_news_data(categories=None, start_date=None, end_date=None):
    """
    Convenience function to get processed news data
    
    Args:
        categories: List of categories
        start_date: Start date
        end_date: End date
    
    Returns:
        Processed DataFrame from MongoDB
    """
    client = MongoClient(
        host=MONGO_CONFIG['host'],
        port=MONGO_CONFIG['port'],
        username=MONGO_CONFIG['username'],
        password=MONGO_CONFIG['password'],
        authSource='admin'
    )
    
    db = client[MONGO_CONFIG['database']]
    collection = db.news_articles_processed
    
    query = {}
    
    if categories:
        query["category"] = {"$in": categories}
    
    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            query["timestamp"]["$gte"] = start_date
        if end_date:
            query["timestamp"]["$lte"] = end_date
    
    cursor = collection.find(query).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))
    
    return df

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = NewsDataPreprocessor()
    
    # Process all available news
    df = preprocessor.run_full_preprocessing()
    
    # Show sample
    if not df.empty:
        print("\n" + "="*80)
        print("PROCESSED NEWS SAMPLE")
        print("="*80)
        print(df[['timestamp', 'category', 'title', 'keywords']].head(10))
        print("\n" + "="*80)
        print("ARTICLES BY CATEGORY")
        print("="*80)
        print(df['category'].value_counts())