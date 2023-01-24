# Database.py
import sqlite3

def connect_to_database(db_file):
    # connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # create the conversation and aliases table
    c.execute('''CREATE TABLE IF NOT EXISTS conversation (input_text TEXT, reply_text TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS aliases (discord_id INTEGER, alias TEXT)''')

    return conn