import sqlite3 
from datetime import datetime 
 
class ParkingLog: 
 
    def __init__(self, db_path="parking.db"): 
 
        self.connection = sqlite3.connect( 
            db_path, 
            check_same_thread=False 
        ) 
 
        self.create_table() 
 
    def create_table(self): 
 
        self.connection.execute(""" 
            CREATE TABLE IF NOT EXISTS detections ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                plate TEXT NOT NULL, 
                country TEXT, 
                confidence REAL, 
                timestamp TEXT NOT NULL, 
                image_path TEXT 
            ) 
        """) 
 
        self.connection.commit() 
 
    def insert( 
        self, 
        plate, 
        country, 
        confidence, 
        image_path=None 
    ): 
 
        timestamp = datetime.now().isoformat() 
 
        self.connection.execute( 
            """ 
            INSERT INTO detections 
            (plate, country, confidence, timestamp, image_path) 
            VALUES (?, ?, ?, ?, ?) 
            """, 
            ( 
                plate, 
                country, 
                confidence, 
                timestamp, 
                image_path 
            ) 
        ) 
 
        self.connection.commit()