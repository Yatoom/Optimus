from pymongo import MongoClient

# Connect to database
def connect():
    db = MongoClient('mongodb://optimus:optimus1@ds213053.mlab.com:13053/optimus_2018')
    table = db.optimus_2018.Lisa
    return db, table

db, table = connect()
table.insert({"foo": "bar"})
