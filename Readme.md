# ☁️ Cloud Cost Optimizer (CCO)

A real-world, highly dynamic Reinforcement Learning (RL) environment built for the **OpenEnv** ecosystem. 

## 📖 Description & Motivation (Real-World Utility)
In the real world, DevOps engineers and SREs (Site Reliability Engineers) constantly battle between two extremes: **Over-provisioning** (wasting thousands of dollars on empty servers) and **Under-provisioning** (crashing the application during traffic spikes). 

The **Cloud Cost Optimizer** simulates this exact enterprise dilemma. An AI agent is tasked with dynamically scaling a fleet of cloud servers up or down in real-time based on incoming web traffic and CPU utilization. To succeed, the agent must learn to maintain a "Golden Ratio" of server capacity—leaving enough of a safety buffer to handle spikes, without bleeding company money on idle compute.

---

## 🛠️ Environment Design: Spaces & Rewards

### 1. Observation Space (`state`)
At every step, the environment provides vital system telemetry:
* `current_traffic` (int): The number of active requests hitting the load balancer.
* `active_instances` (int): The number of currently running server instances.
* `cpu_utilization` (float): The average load across all instances (0.0 to 1.0+).

### 2. Action Space (`step`)
The agent must reply with a strict JSON structure dictating the scaling action:
* `action_type` (string): `"SCALE_UP"`, `"SCALE_DOWN"`, or `"NO_OP"`.
* `instance_count` (int): The absolute number of instances to add or remove (must be >= 0).

### 3. Reward Shaping
The grading mechanism provides dense, continuous feedback over the trajectory:
* **Positive Reward (+0.5 to +1.0):** Maintaining CPU utilization near the optimal threshold (~60-75%). 
* **Negative Penalty (-0.1 to -1.0+):** Overheating servers (CPU > 90%), dropping traffic, or hoarding idle servers.
* **Episode Termination:** The episode finishes after a set number of time-steps, or terminates early with a massive penalty if active servers hit 0 while traffic exists.

---

## 🎯 Task Progression (Agent Graders)
The environment supports three distinct tasks with fully deterministic graders (scores 0.0 - 1.0):
1. **Easy (`easy`)**: Flat, predictable traffic. Tests if the agent can find the baseline CPU efficiency and stop wasting resources.
2. **Medium (`medium`)**: Gradual sinusoidal waves and minor spikes. Tests the agent's ability to smoothly scale up and down without oscillations.
3. **Hard (`hard`)**: Extreme, volatile traffic spikes (e.g., a viral marketing event). Genuinely challenges frontier models to aggressively scale without entering a destructive feedback loop of panic-buying and mass-firing servers.

---

## 🚀 Setup & Execution Steps for Judges

### Option A: Containerized Execution (Docker)
This environment is fully containerized and deployable.
1. Build the image: `docker build -t cloud-cost-optimizer .`
2. Run the container: `docker run -p 7860:7860 cloud-cost-optimizer`

### Option B: Local Terminal Execution
1. **Install Dependencies:**
```bash
uv venv
source .venv/bin/activate
uv pip install -e .