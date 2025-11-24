# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym
import ray


def _default_cases():
    return [
        {
            "description": "Patient reports fever, cough, and shortness of breath after recent travel.",
            "diagnosis": "pneumonia",
            "tools": ["zoom-in", "lab_orders", "vitals_monitor"],
        },
        {
            "description": "Patient complains of severe headache with photophobia and neck stiffness.",
            "diagnosis": "meningitis",
            "tools": ["sam2", "lumbar_puncture", "ct_head"],
        },
        {
            "description": "Patient presents with chest pain radiating to left arm, diaphoresis, and nausea.",
            "diagnosis": "myocardial infarction",
            "tools": ["ecg", "troponin", "biomedparse"],
        },
    ]


class MedicalAgentWorker:
    """Ray worker that hosts one medical environment instance."""

    def __init__(self, seed: int, env_kwargs: dict | None = None):
        self._rng = np.random.RandomState(seed)
        self.env_kwargs = env_kwargs or {}
        self._cases = list(self.env_kwargs.get("cases", _default_cases()))
        if len(self._cases) == 0:
            self._cases = _default_cases()
        self.max_steps = int(self.env_kwargs.get("max_steps", 4))

        self._current_case = None
        self._step_count = 0

    def _sample_case(self):
        idx = self._rng.randint(0, len(self._cases))
        self._current_case = self._cases[idx]
        self._step_count = 0

    def reset(self):
        self._sample_case()
        obs = self._current_case["description"]
        info = {
            "available_tools": list(self._current_case.get("tools", [])),
            "ground_truth": self._current_case.get("diagnosis", ""),
            "won": False,
            "step_count": self._step_count,
        }
        return obs, info

    def step(self, action: str):
        self._step_count += 1
        action_l = action.lower()
        target = self._current_case.get("diagnosis", "").lower()

        won = bool(target and target in action_l)
        reward = 10.0 if won else 0.0
        done = bool(won or self._step_count >= self.max_steps)

        obs = self._current_case["description"]
        info = {
            "available_tools": list(self._current_case.get("tools", [])),
            "ground_truth": self._current_case.get("diagnosis", ""),
            "won": won,
            "step_count": self._step_count,
        }
        return obs, reward, done, info

    def close(self):
        return


class MedicalAgentEnvs(gym.Env):
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train

        env_worker = ray.remote(**resources_per_worker)(MedicalAgentWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), env_kwargs)
            self._workers.append(worker)

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} actions, got {len(actions)}",
            )

        futures = []
        for worker, action in zip(self._workers, actions):
            futures.append(worker.step.remote(action))
        results = ray.get(futures)

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        futures = []
        for worker in self._workers:
            futures.append(worker.reset.remote())
        results = ray.get(futures)

        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for worker in self._workers:
            ray.kill(worker)
        self._closed = True

    def __del__(self):
        self.close()


def build_medical_agent_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict | None = None,
):
    return MedicalAgentEnvs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
