import json, os, glob
from database import get_connection, init_db

def carica_gs():
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    
    gs_path = "/app/gs_data"
    files = glob.glob(os.path.join(gs_path, "*.json"))
    
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data.get("gold_standard", []):
            url = item.get("url", "").rstrip("/")
            
            cur.execute("""
                INSERT IGNORE INTO web_resources (url, domain, title, html_text)
                VALUES (?, ?, ?, ?)
            """, (url, item.get("domain", ""), item.get("title", ""), item.get("html_text", "")))
            
            cur.execute("""
                INSERT IGNORE INTO gold_standard (url, gold_text)
                VALUES (?, ?)
            """, (url, item.get("gold_text", "")))
    
    conn.commit()
    cur.close()
    conn.close()
    print("DB popolato con HTML e GS!")

if __name__ == "__main__":
    carica_gs()