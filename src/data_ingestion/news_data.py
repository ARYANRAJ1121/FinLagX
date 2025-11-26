# src/data_ingestion/news_data.py

import os
import yaml
import feedparser
import pandas as pd
from datetime import datetime
from dateutil import parser as date_parser
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import logging
import hashlib

# Set up logging
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

def get_mongo_client():
    """Get MongoDB client connection"""
    try:
        client = MongoClient(
            host=MONGO_CONFIG['host'],
            port=MONGO_CONFIG['port'],
            username=MONGO_CONFIG['username'],
            password=MONGO_CONFIG['password'],
            authSource='admin'
        )
        # Test connection
        client.admin.command('ismaster')
        return client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

def get_news_collection():
    """Get news collection from MongoDB"""
    client = get_mongo_client()
    db = client[MONGO_CONFIG['database']]
    return db.news_articles

def load_config(path="configs/config_news.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_article_date(date_str):
    """Parse various date formats from RSS feeds"""
    if not date_str:
        return datetime.now()

    try:
        return date_parser.parse(date_str)
    except:
        return datetime.now()

def generate_article_id(title, url):
    """Generate unique article ID from title and URL"""
    content = f"{title}{url}".encode('utf-8')
    return hashlib.md5(content).hexdigest()

def clean_text(text, max_length=None):
    """Clean and truncate text"""
    if not text:
        return ""

    # Basic cleaning
    text = text.strip()
    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text

def fetch_rss_feed(url, category, source_name, limit=50):
    """Fetch RSS feed and return as list of dicts"""
    try:
        feed = feedparser.parse(url)
        articles = []

        for entry in feed.entries[:limit]:
            article_time = parse_article_date(entry.get("published", ""))
            article_id = generate_article_id(
                entry.get("title", ""),
                entry.get("link", "")
            )

            # Create flexible document structure
            article = {
                "_id": article_id,  # MongoDB will use this as primary key
                "article_id": article_id,  # Keep for compatibility
                "timestamp": article_time,
                "title": clean_text(entry.get("title", ""), 500),
                "summary": clean_text(entry.get("summary", ""), 2000),
                "content": clean_text(entry.get("content", "")),  # Some feeds have full content
                "url": entry.get("link", ""),
                "source": {
                    "name": source_name,
                    "url": url,
                    "category": category
                },
                "metadata": {
                    "author": entry.get("author", ""),
                    "tags": entry.get("tags", []),
                    "published_parsed": entry.get("published_parsed"),
                    "language": "en",  # Default, can be detected later
                    "word_count": len(entry.get("summary", "").split()) if entry.get("summary") else 0
                },
                "analysis": {
                    "sentiment_score": None,  # To be calculated later
                    "entities": [],  # Named entities to be extracted
                    "keywords": [],  # Keywords to be extracted
                    "relevance_scores": {}  # Relevance to different asset classes
                },
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }

            articles.append(article)

        return articles

    except Exception as e:
        logger.error(f"Error fetching RSS feed {url}: {e}")
        return []

def save_articles_to_mongo(articles, collection):
    """Save articles to MongoDB with upsert logic"""
    if not articles:
        return 0

    saved_count = 0
    updated_count = 0

    try:
        for article in articles:
            try:
                # Try to insert new article
                result = collection.insert_one(article)
                saved_count += 1
            except DuplicateKeyError:
                # Article exists, update timestamp and metadata
                collection.update_one(
                    {"_id": article["_id"]},
                    {
                        "$set": {
                            "updated_at": datetime.now(),
                            "source": article["source"],  # Update source info
                            "metadata": article["metadata"]
                        }
                    }
                )
                updated_count += 1

        logger.info(f"  Saved {saved_count} new articles, updated {updated_count} existing")
        return saved_count

    except Exception as e:
        logger.error(f"  Error saving articles to MongoDB: {e}")
        return 0

def download_all_news():
    """Download all news and store in MongoDB"""
    config = load_config()
    collection = get_news_collection()

    # Create indexes for better performance
    collection.create_index("timestamp")
    collection.create_index("source.category")
    collection.create_index([("title", "text"), ("summary", "text")])

    logger.info("📰 Starting News Collection Pipeline with MongoDB...\n")

    total_saved = 0
    for category, feeds in config.items():
        logger.info(f"📂 Category: {category}")
        for feed in feeds:
            try:
                articles = fetch_rss_feed(feed["url"], category, feed["name"])
                saved = save_articles_to_mongo(articles, collection)
                total_saved += saved
            except Exception as e:
                logger.error(f"  Error processing {feed['name']}: {e}")

    logger.info(f"\n  News pipeline finished. Total new articles: {total_saved}")

def get_news_data(category=None, start_date=None, end_date=None, limit=100):
    """Query news data from MongoDB"""
    collection = get_news_collection()

    # Build query
    query = {}

    if category:
        query["source.category"] = category

    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            query["timestamp"]["$gte"] = start_date
        if end_date:
            query["timestamp"]["$lte"] = end_date

    # Execute query
    cursor = collection.find(query).sort("timestamp", -1).limit(limit)

    # Convert to DataFrame
    articles = list(cursor)
    if articles:
        return pd.DataFrame(articles)
    else:
        return pd.DataFrame()

def search_news_by_keywords(keywords, category=None, limit=50):
    """Search news by keywords using text index"""
    collection = get_news_collection()

    # Build text search query
    search_text = " ".join(keywords)
    query = {"$text": {"$search": search_text}}

    if category:
        query["source.category"] = category

    # Execute search with text score
    cursor = collection.find(
        query,
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)

    articles = list(cursor)
    if articles:
        return pd.DataFrame(articles)
    else:
        return pd.DataFrame()

def get_news_stats():
    """Get news collection statistics using MongoDB aggregation"""
    collection = get_news_collection()

    pipeline = [
        {
            "$group": {
                "_id": "$source.category",
                "article_count": {"$sum": 1},
                "earliest_date": {"$min": "$timestamp"},
                "latest_date": {"$max": "$timestamp"},
                "avg_word_count": {"$avg": "$metadata.word_count"}
            }
        },
        {"$sort": {"article_count": -1}}
    ]

    stats = list(collection.aggregate(pipeline))
    return pd.DataFrame(stats) if stats else pd.DataFrame()

def update_sentiment_analysis(article_id, sentiment_score, entities=None, keywords=None):
    """Update article with sentiment analysis results"""
    collection = get_news_collection()

    update_doc = {
        "analysis.sentiment_score": sentiment_score,
        "updated_at": datetime.now()
    }

    if entities:
        update_doc["analysis.entities"] = entities
    if keywords:
        update_doc["analysis.keywords"] = keywords

    try:
        result = collection.update_one(
            {"_id": article_id},
            {"$set": update_doc}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error updating sentiment for {article_id}: {e}")
        return False

def get_articles_for_sentiment_analysis(limit=100):
    """Get articles that need sentiment analysis"""
    collection = get_news_collection()

    query = {"analysis.sentiment_score": None}
    cursor = collection.find(query).limit(limit)

    return list(cursor)

def clean_news_collection():
    """Drop the news_articles collection to clear all news data."""
    try:
        collection = get_news_collection()
        collection.drop()
        print("🧹 Cleaned all news data from MongoDB.")
    except Exception as e:
        print(f"  Error cleaning MongoDB collection: {e}")


if __name__ == "__main__":
    logger.info("  Starting News Data Pipeline with MongoDB...\n")

    # Test MongoDB connection
    try:
        client = get_mongo_client()
        logger.info(f"  Connected to MongoDB: {client.server_info()['version']}")
        client.close()
    except Exception as e:
        logger.error(f"  MongoDB connection failed: {e}")
        exit(1)

    # Download all news
    download_all_news()

    # Test queries
    logger.info("\n📊 Testing data retrieval...")

    # Get recent news
    recent_news = get_news_data(limit=5)
    logger.info(f"Recent news: {recent_news.shape}")
    if not recent_news.empty:
        print(recent_news[['timestamp', 'source', 'title']].head())

    # Get news stats
    stats = get_news_stats()
    logger.info(f"\nNews collection statistics:")
    print(stats)

    # Test keyword search
    if not recent_news.empty:
        market_news = search_news_by_keywords(['market', 'stock', 'trading'], limit=3)
        logger.info(f"\nMarket-related news: {len(market_news)} articles")

    logger.info("\n  News data pipeline completed!")