import sqlite3
from datetime import datetime, timezone
import json
import array
from openai import OpenAI
import os, dotenv
from custom_state import State

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
embedding_model = "text-embedding-3-small"

def init_db(path="database/task_history.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS task_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        status TEXT CHECK (status IN ('success','failed')),
        prompt TEXT,
        train_ds_path TEXT,
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

def log_task(conn, state):
    artifacts = create_artifacts(state)
    
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO task_history (created_at, status, prompt, train_ds_path, error, artifacts) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), state.stage, state.prompt, json.dumps(state.train_ds_path), state.error, json.dumps(artifacts))
    )
    task_id = cur.lastrowid
    conn.commit()
    
    vector = build_embedding(state)
    store_embedding(conn, task_id, embedding_model, vector)
    

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


def create_artifacts(state):
    artifacts = {
        "prompt": state.prompt,
        "insights": state.insights,
        "plan": state.plan,
        "preprocess_spec": state.preprocess_spec,
        "training_plan": state.training_plan,
        "all_model_scores": state.all_model_scores
    }
    
    if state.reasoning:
        artifacts["reasoning"] = state.reasoning
    
    return artifacts


def build_embedding(state):
    plan_str = "\n".join([f"{k}: {v}" for k, v in state.plan.items()])
    preprocess_str = "\n".join([f"{k}: {v}" for k, v in state.preprocess_spec.items()])
    training_plan_str = "\n".join([f"{k}: {v}" for k, v in state.training_plan.items()])
    all_model_scores_str = "\n".join([f"{k}: {v}" for k, v in state.all_model_scores.items()])
    
    text = f"Prompt: {state.prompt}\n"
    text += f"Insights: {json.dumps(state.insights)}\n"
    text += f"Plan: \n{plan_str}\n"
    text += f"Preprocess Spec: \n{preprocess_str}\n"
    text += f"Training Plan: \n{training_plan_str}\n"
    text += f"Tested Models with Scores: \n{all_model_scores_str}\n"
    
    if state.reasoning:
        text += f"Reasoning: {state.reasoning}\n"
    
    embedding = client.embeddings.create(
        input=[text],
        model=embedding_model
    )
    vector = embedding.data[0].embedding
    return vector
    

if __name__ == "__main__":
    init_db()
    conn = sqlite3.connect("database/task_history.db")
    state = State(
        prompt="Example prompt",
        raw_train_ds=None,
        fe_train_ds=None,
        train_ds_path="datasets/titanic train.csv",
        insights={"num_rows": 100, "num_columns": 10},
        plan={"task": "regression", "target": "price"},
        preprocess_spec={"feature_engineering": []},
        training_plan={"model_type": "linear_regression"},
        all_model_scores={"linear_regression": {"rmse": 5.0}},
        reasoning="This is an example reasoning.",
        stage="success",
        error=None
    )
    log_task(conn, state)
    result = fetch_task_embeddings(conn, 1)
    print(result)