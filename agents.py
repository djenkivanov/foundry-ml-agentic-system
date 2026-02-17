import pandas as pd
import logic
import sqlite3
import os
import json
import traceback
import db
from custom_state import State

def planner_agent(state: State, reasoning_stream=None, plan_stream=None):
    try:
        logic.get_data_insight(state)
        logic.create_initial_plan(
            state=state,
            reasoning_stream=reasoning_stream,
            plan_stream=plan_stream
        )
    except Exception as e:
        state.error = str(e)
        state.stage = "failed"


def preprocessing_agent(state: State):
    try:
        logic.create_preprocess_spec(state)
        logic.execute_preprocess_spec(state)
    except Exception as e:
        state.error = str(e)
        state.stage = "failed"
        
        
def training_agent(state: State):
    try:
        logic.refine_training_plan(state)
        logic.convert_training_plan_to_code(state)
    except Exception as e:
        state.error = str(e)
        state.stage = "failed"
        

def package_agent(state: State):
    try:
        logic.package_model(state)
    except Exception as e:
        state.error = str(e)
        state.stage = "failed"
        

def self_learning_agent(state: State, task_id: int):
    try:
        db.init_db()
        conn = sqlite3.connect("database/task_history.db")
        cur = conn.cursor()
        
        if task_id is None:
            cur.execute("SELECT id FROM task_history WHERE status = 'failed' ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            
            if not row:
                return None
            
            task_id = row[0]
            
        task = db.fetch_task(conn, task_id)
        
        if not task:
            return None
        
        train_ds_path = task.get("train_ds_path")
        
        if not train_ds_path or not os.path.exists(train_ds_path):
            raise FileNotFoundError(f"Train dataset not found: {train_ds_path}")
        
        df = pd.read_csv(train_ds_path)
        artifacts = task.get("artifacts", {})
        prev_plan = artifacts.get("plan", {})
        prev_prompt = task.get("prompt") or ""
        improved_prompt = "Improve the previous plan for this task. \nPrevious plan: " + json.dumps(prev_plan) + f"\nOriginal prompt: {prev_prompt}"
        
        new_state = State(
            prompt=improved_prompt,
            raw_train_ds=df,
            fe_train_ds=df.copy(),
            train_ds_path=train_ds_path,
            insights=artifacts.get("insights", {}),
            plan=artifacts.get("plan", {}),
            preprocess_spec=artifacts.get("preprocess_spec", {}),
            training_plan=artifacts.get("training_plan", {}),
            all_model_scores=artifacts.get("all_model_scores", {}),
            reasoning=artifacts.get("reasoning", ""),
            trace=artifacts.get("trace", []) or []
        )
        
        if new_state.trace is None:
            new_state.trace = []
            
        new_state.trace.append({"self_learning": {"rerun_of_task_id": task_id}})
        
        try:
            planner_agent(new_state)
            if new_state.stage == "failed":
                raise RuntimeError("Planner failed")
            preprocessing_agent(new_state)
            if new_state.stage == "failed":
                raise RuntimeError("Preprocessing failed")
            training_agent(new_state)
            if new_state.stage == "failed":
                raise RuntimeError("Training failed")
            package_agent(new_state)
            if new_state.stage == "failed":
                raise RuntimeError("Packaging failed")
        except Exception as e:
            new_state.error = str(e)
            new_state.stage = "failed"
            new_state.trace.append({"error": traceback.format_exc()})
            
        try:
            db.log_task(new_state)
        except Exception as e:
            print("Failed to log rerun:", e)
            
        if state is not None:
            for k, v in vars(new_state).items():
                setattr(state, k, v)
                
        return new_state
    
    finally:
        try:
            conn.close()
        except:
            pass
