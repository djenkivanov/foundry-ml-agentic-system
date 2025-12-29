from email.mime import text
import sqlite3
from datetime import datetime, timezone
import json
import array
from openai import OpenAI
import os, dotenv
from custom_state import State
from sklearn.metrics.pairwise import cosine_similarity

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

def log_task(state):
    init_db()
    artifacts = create_artifacts(state)
    
    conn = sqlite3.connect("database/task_history.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO task_history (created_at, status, prompt, train_ds_path, error, artifacts) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), state.stage, state.prompt, state.train_ds_path, state.error, json.dumps(artifacts))
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


def fetch_task(conn, task_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, status, prompt, train_ds_path, error, artifacts FROM task_history WHERE id = ?",
        (task_id,)
    )
    row = cur.fetchone()
    if row:
        task_id, created_at, status, prompt, train_ds_path, error, artifacts_json = row
        artifacts = json.loads(artifacts_json)
        return {
            "id": task_id,
            "created_at": created_at,
            "status": status,
            "prompt": prompt,
            "train_ds_path": train_ds_path,
            "error": error,
            "artifacts": artifacts
        }
    return None


def create_artifacts(state):
    artifacts = {
        "prompt": state.prompt,
        "insights": state.insights,
        "plan": state.plan,
        "preprocess_spec": state.preprocess_spec,
        "training_plan": state.training_plan,
        "all_model_scores": state.all_model_scores,
        "trace": state.trace
    }
    
    if state.reasoning:
        artifacts["reasoning"] = state.reasoning
    
    return artifacts


def build_embedding(state):
    insights_str = "\n".join([f"\t{k}: {v}" for k, v in state.insights.items()])
    plan_str = "\n".join([f"\t{k}: {v}" for k, v in state.plan.items()])
    preprocess_str = "\n".join([f"\t{k}: {v}" for k, v in state.preprocess_spec.items()])
    training_plan_str = "\n".join([f"\t{k}: {v}" for k, v in state.training_plan.items()])
    all_model_scores_str = "\n".join([f"\t{k}: {v}" for k, v in state.all_model_scores.items()])
    trace_str = ""
    for step in state.trace:
        step_str = ", ".join([f"\n\t{k}: {v}" for k, v in step.items()])
        trace_str += f"{step_str}\n"
    
    text = f"Prompt: \n\t{state.prompt}\n"
    text += f"Insights: \n{insights_str}\n"
    text += f"Plan: \n{plan_str}\n"
    text += f"Preprocess Spec: \n{preprocess_str}\n"
    text += f"Training Plan: \n{training_plan_str}\n"
    text += f"Tested Models with Scores: \n{all_model_scores_str}\n"
    text += f"Task Trace: {trace_str}"
    
    if state.reasoning:
        text += f"Reasoning: \n\t{state.reasoning}\n"
    
    print(text)
    
    embedding = client.embeddings.create(
        input=[text],
        model=embedding_model
    )
    vector = embedding.data[0].embedding
    return vector
    

def find_similar_tasks(state, top_k=3):
    plan, reasoning, insights = state.plan, state.reasoning, state.insights
    similarities = []
    most_similar_task_ids = []
    threshold = 0.5

    conn = sqlite3.connect("database/task_history.db")
    cur = conn.cursor()
    cur.execute("SELECT id, artifacts FROM task_history")
    rows = cur.fetchall()
    
    current_task_vector = client.embeddings.create(
        input=[f"Plan: {json.dumps(plan)}\nReasoning: {reasoning}\nInsights: {json.dumps(insights)}"],
        model=embedding_model
    ).data[0].embedding
    
    for task_id, artifacts_json in rows:
        artifacts = json.loads(artifacts_json)
        task_plan = artifacts.get("plan", {})
        task_reasoning = artifacts.get("reasoning", "")
        task_insights = artifacts.get("insights", {})
        
        vector = client.embeddings.create(
            input=[f"Plan: {json.dumps(task_plan)}\nReasoning: {task_reasoning}\nInsights: {json.dumps(task_insights)}"],
            model=embedding_model
        ).data[0].embedding
        
        sim = compute_similarity(current_task_vector, vector)
        
        similarities.append((task_id, sim))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    for task_id, sim in similarities[:top_k]:
        if sim >= threshold:
            most_similar_task_ids.append((task_id, sim))

    task_ids_str = [t_id for t_id, _ in most_similar_task_ids]
    sim_score_map = {task_id: sim for task_id, sim in most_similar_task_ids}

    cur.execute(f"SELECT id, created_at, status, prompt, train_ds_path, error, artifacts FROM task_history WHERE id IN \
        ({','.join(['?']*len(task_ids_str))})", task_ids_str)
    rows = cur.fetchall()
    similar_tasks = []
    for row in rows:
        task_id, created_at, status, prompt, train_ds_path, error, artifacts_json = row
        artifacts = json.loads(artifacts_json)
        similar_tasks.append({
            "id": task_id,
            "created_at": created_at,
            "status": status,
            "prompt": prompt,
            "train_ds_path": train_ds_path,
            "error": error,
            "artifacts": artifacts,
            "similarity_score": sim_score_map.get(task_id)
        })

    return similar_tasks

    
def compute_similarity(vec1, vec2):
    sim = cosine_similarity([vec1], [vec2])[0][0]
    return sim
        

if __name__ == "__main__":
    init_db()
    conn = sqlite3.connect("database/task_history.db")
    
    trace = [
        {"step": "plan", "detail": "Created initial plan."},
        {"step": "preprocess", "detail": "Generated preprocessing specification."},
        {"step": "train", "detail": "Completed model training."}
    ]
    
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
        error=None,
        trace=trace
    )
    log_task(conn, state)
    result = fetch_task_embeddings(conn, 1)
    print(result)
    task = fetch_task(conn, 1)
    print(task)