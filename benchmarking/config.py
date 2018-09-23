from pymongo import MongoClient
import openml

# Setup API key
openml.config.apikey = "YOUR_API_KEY"


# Connect to database
def connect():
    db = MongoClient('mongodb://<URI>')
    table = db.optimus.Benchmark

    return db, table