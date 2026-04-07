import math
from schemas import Action, ActionType, Observation, Reward, StepResponse

class CloudOptimizerEnv:
    def __init__(self):
        # Server physics
        self.capacity_per_server = 100
        self.cost_per_server = 0.1
        self.penalty_per_drop = 5.0
        
        # Task definitions (Traffic profiles over 5 steps)
        self.tasks = {
            "easy": [400, 500, 600, 500, 500],
            "medium": [200, 400, 800, 400, 200],
            "hard": [100, 900, 200, 1000, 100]
        }
        
        self.reset("easy")

    def reset(self, task_id: str = "easy"):
        self.current_task_id = task_id
        self.traffic_profile = self.tasks.get(task_id, self.tasks["easy"])
        self.current_step = 0
        self.active_instances = 2  # Start with 2 servers
        self.total_score = 0.0
        self.max_possible_score = len(self.traffic_profile) * 1.0 # 1.0 max per step
        
        return self._get_observation()

    def _get_observation(self):
        if self.current_step >= len(self.traffic_profile):
            current_traffic = 0
        else:
            current_traffic = self.traffic_profile[self.current_step]
            
        total_capacity = self.active_instances * self.capacity_per_server
        
        # Prevent division by zero if agent shuts down all servers
        if total_capacity == 0:
            cpu_utilization = 1.0 if current_traffic > 0 else 0.0
        else:
            cpu_utilization = current_traffic / total_capacity

        # Exponential latency curve
        if cpu_utilization <= 0.8:
            latency_ms = 50.0
        else:
            latency_ms = 50.0 * math.exp(3 * (cpu_utilization - 0.8))

        return Observation(
            current_traffic=current_traffic,
            active_instances=self.active_instances,
            cpu_utilization=round(cpu_utilization, 2),
            latency_ms=round(latency_ms, 2)
        )

    def step(self, action: Action) -> StepResponse:
        if action.action_type == ActionType.SCALE_UP:
            self.active_instances += action.instance_count
        elif action.action_type == ActionType.SCALE_DOWN:
            self.active_instances = max(0, self.active_instances - action.instance_count)

        obs = self._get_observation()
        
        current_traffic = self.traffic_profile[self.current_step]
        total_capacity = self.active_instances * self.capacity_per_server
        
        dropped_requests = max(0, current_traffic - total_capacity)
        cost_incurred = self.active_instances * self.cost_per_server
        
        step_reward = 1.0 
        
        step_reward -= cost_incurred
        step_reward -= (dropped_requests * self.penalty_per_drop)
        
        step_reward = max(-10.0, min(1.0, step_reward))
        self.total_score += step_reward

        reward_obj = Reward(
            value=round(step_reward, 2),
            dropped_requests=dropped_requests,
            cost_incurred=round(cost_incurred, 2)
        )

        self.current_step += 1
        done = self.current_step >= len(self.traffic_profile)

        return StepResponse(
            observation=obs,
            reward=reward_obj,
            done=done,
            info={"task": self.current_task_id, "step": self.current_step}
        )

    def get_grader_score(self) -> float:
        # Normalize the final score to a 0.0 - 1.0 range
        normalized = max(0.0, self.total_score / self.max_possible_score)
        return round(normalized, 2)