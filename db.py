import sqlite3
from datetime import datetime, timezone
import json
import array
from openai import OpenAI
import os, dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def init_db(path="database/task_history.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS task_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        status TEXT CHECK (status IN ('success','failed')),
        user_prompt TEXT,
        train_ds TEXT,
        error TEXT,
        artifacts JSON
    );
    CREATE TABLE IF NOT EXISTS task_embeddings (
        task_id INTEGER NOT NULL,
        model TEXT NOT NULL,
        dim INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        FOREIGN KEY(task_id) REFERENCES task_history(id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    conn.close()

def log_task(conn, status, user_prompt, train_ds, error, artifacts):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO task_history (created_at, status, user_prompt, train_ds, error, artifacts) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), status, user_prompt, json.dumps(train_ds), error, json.dumps(artifacts))
    )
    task_id = cur.lastrowid
    conn.commit()
    return task_id

def store_embedding(conn, task_id, model, vector):
    buf = array.array('f', vector).tobytes()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO task_embeddings (task_id, model, dim, embedding) VALUES (?, ?, ?, ?)",
        (task_id, model, len(vector), buf)
    )
    conn.commit()
    

def fetch_task_embeddings(conn, task_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT model, dim, embedding FROM task_embeddings WHERE task_id = ?",
        (task_id,)
    )
    rows = cur.fetchall()
    embeddings = []
    for model, dim, blob in rows:
        vector = array.array('f')
        vector.frombytes(blob)
        embeddings.append((model, dim, vector.tolist()))
    return embeddings

    
if __name__ == "__main__":
    init_db()
    conn = sqlite3.connect("database/task_history.db")
    # task_id = log_task(conn, "success", "", {"train_dataset": "sample"}, "Out of memory", {"artifact_1": "value_1"})
    # embedding_trace = client.embeddings.create(
    #     input=["This is a sample embedding."],
    #     model="text-embedding-3-small"
    # )
    # vector = embedding_trace.data[0].embedding
    # store_embedding(conn, task_id, "text-embedding-3-small", vector)
    embeddings = fetch_task_embeddings(conn, 1)
    print(embeddings)