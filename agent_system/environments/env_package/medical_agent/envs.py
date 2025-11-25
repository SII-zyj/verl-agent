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

import gym
import numpy as np
import ray
import re


DEFAULT_TOOLS = ["Zoom-in", "BiomedParse", "SAM2"]


def _default_cases():
    return [
        {
            "question": "Based on this CT image, is there an abnormality?",
            "question_type": "multiple choice",
            "options": [
                "(A) pancreas tumor",
                "(B) No abnormality detected",
                "(C) liver tumor",
                "(D) kidney tumor",
                "(E) kidney cyst",
            ],
            "answer": "C",
            "data_type": "vqa",
            "source": "biomedparse",
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

        self._current_case: dict | None = None
        self._step_count = 0
        self._data_type = "vqa"
        self.max_steps = int(self.env_kwargs.get("max_steps", 4))

    def _sample_case(self):
        idx = self._rng.randint(0, len(self._cases))
        return self._cases[idx]

    def _format_overview(self, case: dict) -> str:
        question = case.get("question", "")
        options = case.get("options") or []
        if isinstance(options, (list, tuple)) and len(options) > 0:
            options_block = "\n" + "\n".join(str(opt) for opt in options)
        else:
            options_block = ""
        question_type = case.get("question_type")
        if question_type:
            prefix = f"[{question_type}] "
        else:
            prefix = ""
        return f"{prefix}{question}{options_block}"

    def _prepare_case(self, env_kwargs: dict | None) -> dict:
        if env_kwargs:
            return env_kwargs
        return self._sample_case()

    def reset(self, env_kwargs: dict | None = None):
        case = self._prepare_case(env_kwargs)
        self._current_case = case
        self._step_count = 0
        self._data_type = (case.get("data_type") or "vqa").lower()
        self.max_steps = int(case.get("max_steps", self.max_steps))

        obs = self._format_overview(case)
        info = {
            "available_tools": list(case.get("available_tools", DEFAULT_TOOLS)),
            "ground_truth": case.get("answer", ""),
            "data_type": case.get("data_type"),
            "question_id": case.get("question_id"),
            "question_type": case.get("question_type"),
            "source": case.get("source"),
            "image_path": case.get("image_path"),
            "mask_path": case.get("mask_path"),
            "won": False,
            "step_count": self._step_count,
        }
        return obs, info

    def _extract_answer(self, action: str) -> str | None:
        for pattern in [r"<answer>\s*([A-Za-z])\b", r"answer\s*[:：]\s*([A-Za-z])"]:
            match = re.search(pattern, action, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().upper()
        # fallback: try to find standalone option letter
        letter_match = re.search(r"\b([A-E])\b", action, flags=re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        return None

    def step(self, action: str):
        self._step_count += 1
        obs = self._format_overview(self._current_case)

        predicted_answer = self._extract_answer(action) or ""
        target_answer = (self._current_case.get("answer") or "").upper()

        won = bool(predicted_answer and target_answer and predicted_answer == target_answer)
        reward = 1.0 if won else 0.0
        done = bool(won or self._step_count >= self.max_steps)

        info = {
            "available_tools": list(self._current_case.get("available_tools", DEFAULT_TOOLS)),
            "ground_truth": target_answer,
            "data_type": self._data_type,
            "question_id": self._current_case.get("question_id"),
            "question_type": self._current_case.get("question_type"),
            "source": self._current_case.get("source"),
            "image_path": self._current_case.get("image_path"),
            "mask_path": self._current_case.get("mask_path"),
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
        if len(actions) > self.num_processes:
            raise ValueError(
                f"Expected at most {self.num_processes} actions, got {len(actions)}",
            )

        pad_n = self.num_processes - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        futures = []
        for worker, action in zip(self._workers, padded_actions):
            futures.append(worker.step.remote(action))
        results = ray.get(futures)

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for keep, result in zip(valid_mask, results):
            if not keep:
                continue
            obs, reward, done, info = result
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self, kwargs: list[dict] | None = None):
        kwargs = kwargs or []
        if len(kwargs) > self.num_processes:
            raise ValueError(
                f"Expected at most {self.num_processes} env kwargs, got {len(kwargs)}",
            )

        pad_n = self.num_processes - len(kwargs)
        padded_kwargs = list(kwargs) + [{}] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        futures = []
        for worker, env_kwargs in zip(self._workers, padded_kwargs):
            futures.append(worker.reset.remote(env_kwargs))
        results = ray.get(futures)

        obs_list, info_list = [], []
        for keep, result in zip(valid_mask, results):
            if not keep:
                continue
            obs, info = result
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
