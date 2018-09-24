from benchmarking.config import connect

db, table = connect()
table.insert({"foo": "bar"})