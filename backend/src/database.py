import mariadb
import time
import os

DB_CONFIG = {
    "host": "mariadb",
    "port": 3306,
    "user": "parser",
    "password": "parserpass",
    "database": "parserdb"
}

def get_connection(retries=10, delay=5):  
    for i in range(retries):
        try:
            return mariadb.connect(**DB_CONFIG, autocommit=True) 
        except mariadb.Error as e:
            if i < retries - 1:
                print(f"DB non pronto, riprovo in {delay}s... ({e})")
                time.sleep(delay)
            else:
                raise

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS web_resources (
            url VARCHAR(500) PRIMARY KEY,
            domain VARCHAR(200),
            title TEXT,
            html_text LONGTEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gold_standard (
            url VARCHAR(500) PRIMARY KEY,
            gold_text LONGTEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url) REFERENCES web_resources(url) ON DELETE CASCADE
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS parsed_documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            url VARCHAR(500),
            parsed_text LONGTEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url) REFERENCES web_resources(url) ON DELETE CASCADE
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            url VARCHAR(500),
            precision_score FLOAT,
            recall_score FLOAT,
            f1_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url) REFERENCES web_resources(url) ON DELETE CASCADE
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_judgments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            url VARCHAR(500),
            model_name VARCHAR(100),
            judge_score INT,
            judge_feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url) REFERENCES web_resources(url) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()
