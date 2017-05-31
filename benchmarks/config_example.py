from pymongo import MongoClient


def connect():
    db = MongoClient('mongodb://<URI>')
    table = db.optimus.Benchmark

    return db, table
