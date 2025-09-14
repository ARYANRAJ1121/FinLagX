db = db.getSiblingDB('finlagx_news');

db.createCollection("news_articles", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["article_id", "timestamp", "title", "source"],
         properties: {
            article_id: {
               bsonType: "string",
               description: "Unique identifier for the article"
            },
            timestamp: {
               bsonType: "date",
               description: "Publication timestamp"
            },
            title: {
               bsonType: "string",
               maxLength: 500,
               description: "Article title"
            },
            summary: {
               bsonType: "string",
               maxLength: 2000,
               description: "Article summary"
            },
            url: {
               bsonType: "string",
               description: "Article URL"
            },
            source: {
               bsonType: "object",
               required: ["name", "category"],
               properties: {
                  name: { bsonType: "string" },
                  url: { bsonType: "string" },
                  category: { bsonType: "string" }
               }
            },
            analysis: {
               bsonType: "object",
               properties: {
                  sentiment_score: { bsonType: ["double", "null"] },
                  entities: { bsonType: "array" },
                  keywords: { bsonType: "array" },
                  relevance_scores: { bsonType: "object" }
               }
            }
         }
      }
   }
});

// Create indexes for optimal query performance
db.news_articles.createIndex({ "timestamp": -1 });
db.news_articles.createIndex({ "source.category": 1 });
db.news_articles.createIndex({ "source.name": 1 });
db.news_articles.createIndex({ "title": "text", "summary": "text" });
db.news_articles.createIndex({ "analysis.sentiment_score": 1 });
db.news_articles.createIndex({ "url": 1 }, { unique: true });

// Create compound indexes for common queries
db.news_articles.createIndex({ "source.category": 1, "timestamp": -1 });
db.news_articles.createIndex({ "analysis.sentiment_score": 1, "timestamp": -1 });

// Insert sample document to test schema
db.news_articles.insertOne({
   "article_id": "sample_001",
   "timestamp": new Date(),
   "title": "Sample Financial News Article",
   "summary": "This is a sample news article for testing the MongoDB schema.",
   "url": "https://example.com/sample-article",
   "source": {
      "name": "sample_source",
      "url": "https://example.com/rss",
      "category": "equities"
   },
   "metadata": {
      "author": "Sample Author",
      "word_count": 50,
      "language": "en"
   },
   "analysis": {
      "sentiment_score": null,
      "entities": [],
      "keywords": [],
      "relevance_scores": {}
   },
   "created_at": new Date(),
   "updated_at": new Date()
});

db.createUser({
   user: "finlagx_app",
   pwd: "finlagx_news_password",
   roles: [
      {
         role: "readWrite",
         db: "finlagx_news"
      }
   ]
});

print("✅ FinLagX News Database initialized successfully!");
print("📊 Created indexes and sample data");
print("👤 Created application user: finlagx_app");