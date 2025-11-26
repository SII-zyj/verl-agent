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

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
import ray
import requests
from PIL import Image

from agent_system.environments.prompts.medical_agent import (
    MEDICAL_AGENT_SYSTEM_PROMPT,
    MEDICAL_AGENT_TOOL_FEEDBACK,
    MEDICAL_AGENT_USER_PROMPT,
)


DEFAULT_TOOLS = ["Zoom-in", "BiomedParse", "SAM2"]
DEFAULT_TOOL_ENDPOINTS = {
    "BiomedParse": "/biomedparse",
    "SAM2": "/sam2",
}


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
            "image_path": [],
        },
    ]


def _read_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.width, img.height


def _load_image_array(path: str) -> np.ndarray:
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        return np.array(rgb_img)


class MedicalToolClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def run(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Tool server must return a JSON object.")
        return data


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

        self.output_dir = Path(self.env_kwargs.get("output_dir", "/tmp/medical_agent"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_server_url = self.env_kwargs.get("tool_server_url", "http://127.0.0.1:6006")
        self.tool_timeout = int(self.env_kwargs.get("tool_timeout", 30))
        self.tool_endpoints = {
            **DEFAULT_TOOL_ENDPOINTS,
            **self.env_kwargs.get("tool_endpoints", {}),
        }
        self.tool_client = MedicalToolClient(self.tool_server_url, self.tool_timeout)

        self._messages: list[dict[str, str]] = []
        self._images: list[dict[str, Any]] = []

    def _sample_case(self):
        idx = self._rng.randint(0, len(self._cases))
        return self._cases[idx]

    def _prepare_case(self, env_kwargs: dict | None) -> dict:
        if env_kwargs:
            return env_kwargs
        return self._sample_case()

    def _format_options(self, options: list | tuple | None) -> str:
        if not options:
            return ""
        return "\n".join(str(opt) for opt in options)

    def _resolve_image_path(self, case: dict) -> str:
        image_path = case.get("image_path") or []
        if isinstance(image_path, (list, tuple)):
            if len(image_path) == 0:
                raise ValueError("image_path is required for medical agent cases.")
            resolved = image_path[0]
        else:
            resolved = image_path
        resolved = os.path.abspath(resolved)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Image not found at {resolved}")
        return resolved

    def _register_image(self, path: str, mask_path: str | None = None) -> dict:
        width, height = _read_image_size(path)
        meta = {
            "index": len(self._images) + 1,
            "path": path,
            "width": width,
            "height": height,
            "mask_path": mask_path,
        }
        self._images.append(meta)
        return meta

    def _build_user_prompt(self, case: dict, image_meta: dict) -> str:
        options_block = self._format_options(case.get("options"))
        return MEDICAL_AGENT_USER_PROMPT.format(
            question=case.get("question", ""),
            options=options_block,
            image_index=image_meta["index"],
            width=image_meta["width"],
            height=image_meta["height"],
        ).strip()

    def _format_tool_feedback(self, image_meta: dict) -> str:
        return MEDICAL_AGENT_TOOL_FEEDBACK.format(
            image_index=image_meta["index"],
            width=image_meta["width"],
            height=image_meta["height"],
        ).strip()

    def _compose_observation(self) -> Tuple[str, np.ndarray]:
        text_obs = "\n\n".join(message["content"] for message in self._messages)
        latest_image = self._images[-1]["path"] if self._images else None
        image_obs = _load_image_array(latest_image) if latest_image else None
        return text_obs, image_obs

    def reset(self, env_kwargs: dict | None = None):
        case = self._prepare_case(env_kwargs)
        self._current_case = case
        self._step_count = 0
        self._data_type = (case.get("data_type") or "vqa").lower()
        self.max_steps = int(case.get("max_steps", self.max_steps))

        self._messages = []
        self._images = []

        initial_image_path = self._resolve_image_path(case)
        image_meta = self._register_image(initial_image_path, case.get("mask_path"))

        system_prompt = MEDICAL_AGENT_SYSTEM_PROMPT.strip()
        user_prompt = self._build_user_prompt(case, image_meta)
        self._messages.append({"role": "system", "content": system_prompt})
        self._messages.append({"role": "user", "content": user_prompt})

        text_obs, image_obs = self._compose_observation()
        info = self._build_info(won=False)
        return text_obs, image_obs, info

    def _build_info(self, won: bool) -> dict:
        latest_mask = self._images[-1].get("mask_path") if self._images else None
        return {
            "available_tools": list(self._current_case.get("available_tools", DEFAULT_TOOLS)),
            "ground_truth": (self._current_case.get("answer") or "").upper(),
            "data_type": self._data_type,
            "question_id": self._current_case.get("question_id"),
            "question_type": self._current_case.get("question_type"),
            "source": self._current_case.get("source"),
            "image_path": [img["path"] for img in self._images],
            "mask_path": latest_mask,
            "won": won,
            "step_count": self._step_count,
        }

    def _extract_answer(self, action: str) -> str | None:
        for pattern in [r"<answer>\s*([A-Za-z])\b", r"answer\s*[:：]\s*([A-Za-z])"]:
            match = re.search(pattern, action, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().upper()
        letter_match = re.search(r"\b([A-E])\b", action, flags=re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        return None

    def _get_image_by_index(self, index: int) -> dict:
        if index <= 0 or index > len(self._images):
            raise IndexError(f"Image index {index} is out of bounds.")
        return self._images[index - 1]

    def _parse_tool_invocations(self, action: str) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        if not action:
            return actions
        for tool in DEFAULT_TOOLS:
            pattern = rf"{tool}.*?```json\s*(.*?)\s*```"
            match = re.search(pattern, action, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                pattern = rf"<tool_call>\s*{tool}\s*```json\s*(.*?)\s*```"
                match = re.search(pattern, action, flags=re.IGNORECASE | re.DOTALL)
            if match:
                payload_str = match.group(1)
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    payload = [payload]
                if isinstance(payload, list):
                    actions.append({"tool": tool, "payload": payload})
        return actions

    def _call_segmenter(self, tool: str, image_path: str, extra_payload: dict) -> str | None:
        endpoint = self.tool_endpoints.get(tool)
        if not endpoint:
            raise ValueError(f"No endpoint configured for tool {tool}")
        payload = {"image": image_path}
        payload.update(extra_payload)
        response = self.tool_client.run(endpoint, payload)
        return response.get("mask_path") or response.get("image_path")

    def _handle_biomedparse(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        captions = payload.get("captions", "")
        base_image = self._get_image_by_index(idx)
        mask_path = self._call_segmenter("BiomedParse", base_image["path"], {"captions": captions})
        if not mask_path:
            raise RuntimeError("BiomedParse did not return a mask path.")
        mask_path = os.path.abspath(mask_path)
        image_meta = self._register_image(mask_path, mask_path)
        return image_meta

    def _handle_sam2(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("SAM2 requires bbox_2d with four coordinates.")
        base_image = self._get_image_by_index(idx)
        mask_path = self._call_segmenter("SAM2", base_image["path"], {"bbox_2d": bbox})
        if not mask_path:
            raise RuntimeError("SAM2 did not return a mask path.")
        mask_path = os.path.abspath(mask_path)
        image_meta = self._register_image(mask_path, mask_path)
        return image_meta

    def _clamp_bbox(self, bbox: list[float], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), width))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height))
        y2 = max(0, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box after clamping.")
        return x1, y1, x2, y2

    def _handle_zoom_in(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("Zoom-in requires bbox_2d with four coordinates.")
        base_image = self._get_image_by_index(idx)
        x1, y1, x2, y2 = self._clamp_bbox(bbox, base_image["width"], base_image["height"])
        with Image.open(base_image["path"]) as img:
            cropped = img.crop((x1, y1, x2, y2))
            new_name = f"zoom_{uuid.uuid4().hex}.png"
            new_path = str(self.output_dir / new_name)
            cropped.save(new_path)
        image_meta = self._register_image(new_path, base_image.get("mask_path"))
        return image_meta

    def _execute_tool_action(self, tool: str, payload: dict) -> dict:
        if tool == "BiomedParse":
            return self._handle_biomedparse(payload)
        if tool == "SAM2":
            return self._handle_sam2(payload)
        if tool == "Zoom-in":
            return self._handle_zoom_in(payload)
        raise ValueError(f"Unsupported tool: {tool}")

    def step(self, action: str):
        self._step_count += 1
        action = action or ""
        self._messages.append({"role": "assistant", "content": action})

        reward = 0.0
        done = False
        won = False
        new_feedback: list[str] = []

        tool_invocations = self._parse_tool_invocations(action)
        if tool_invocations:
            for item in tool_invocations:
                tool_name = item["tool"]
                for payload in item.get("payload", []):
                    try:
                        image_meta = self._execute_tool_action(tool_name, payload)
                        feedback = self._format_tool_feedback(image_meta)
                        new_feedback.append(feedback)
                    except Exception as exc:  # noqa: BLE001
                        new_feedback.append(str(exc))
        else:
            predicted_answer = self._extract_answer(action) or ""
            target_answer = (self._current_case.get("answer") or "").upper()
            won = bool(predicted_answer and target_answer and predicted_answer == target_answer)
            reward = 1.0 if won else 0.0
            done = bool(predicted_answer)

        if new_feedback:
            for feedback in new_feedback:
                self._messages.append({"role": "user", "content": feedback})

        if self._step_count >= self.max_steps:
            done = True

        text_obs, image_obs = self._compose_observation()
        info = self._build_info(won)
        return text_obs, image_obs, reward, done, info

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

        text_list, image_list, reward_list, done_list, info_list = [], [], [], [], []
        for keep, result in zip(valid_mask, results):
            if not keep:
                continue
            obs, image_obs, reward, done, info = result
            text_list.append(obs)
            image_list.append(image_obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return text_list, image_list, reward_list, done_list, info_list

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

        text_list, image_list, info_list = [], [], []
        for keep, result in zip(valid_mask, results):
            if not keep:
                continue
            obs, image_obs, info = result
            text_list.append(obs)
            image_list.append(image_obs)
            info_list.append(info)
        return text_list, image_list, info_list

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
