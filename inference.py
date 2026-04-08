import os
import sys
import requests
import time
import json
import re
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

API_KEY = HF_TOKEN or os.getenv("API_KEY")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

BENCHMARK = "cloud-cost-optimizer"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action(obs):
    traffic = obs['current_traffic']
    active = obs['active_instances']
    
    prompt = f"Traffic: {traffic}, Active: {active}. Output ONLY valid JSON: {{'action_type': '...', 'instance_count': ...}}"
    
    completion = client.chat.completions.create(
        model=MODEL_NAME, 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
        stream=False
    )
    
    raw_text = completion.choices[0].message.content
    match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
    if match:
        return json.loads(match.group(0)), None
        
    return {"action_type": "NO_OP", "instance_count": 0}, "JSON_PARSE_ERROR"

def run_agent(task_id: str):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    server_awake = False
    for _ in range(20): 
        try:
            requests.post(f"{API_URL}/reset", json={"task_id": task_id}, timeout=5)
            server_awake = True
            break
        except Exception:
            time.sleep(2) 
            
    if not server_awake:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return
        
    done = False
    step_count = 0
    rewards_history = []
    
    while not done:
        step_count += 1
        try:
            obs = requests.get(f"{API_URL}/state", timeout=5).json()["observation"]
            
            action_payload, error = get_action(obs)
            action_str = json.dumps(action_payload).replace(" ", "") 
            
            step_res = requests.post(f"{API_URL}/step", json=action_payload, timeout=5).json()
            done = step_res["done"]
            reward = float(step_res['reward']['value'])
            
            rewards_history.append(reward)
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)
            
        except Exception as e:
            print(f"[DEBUG] Proxy Crash: {e}", file=sys.stderr, flush=True)
            log_step(step=step_count, action="null", reward=0.0, done=True, error=str(e)[:50])
            break
            
    try:
        score = float(requests.get(f"{API_URL}/grader", timeout=5).json()['score'])
        success = score >= 0.5 
    except Exception:
        score = 0.0
        success = False
        
    log_end(success=success, steps=step_count, score=score, rewards=rewards_history)

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_agent(task)
