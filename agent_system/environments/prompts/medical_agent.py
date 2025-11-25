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

MEDICAL_AGENT_TEMPLATE_NO_HIS = """
You are a clinical reasoning agent operating in a simulated hospital.
Here is the patient overview: {patient_overview}

Available clinical tools you can call: 
{available_tools}

First provide your differential reasoning inside <think> </think>.
Then present a concrete diagnostic or treatment action inside <action> </action>.
"""

MEDICAL_AGENT_TEMPLATE = """
You are a clinical reasoning agent operating in a simulated hospital.
Here is the patient overview: {patient_overview}
You have taken {step_count} step(s). The most recent {history_length} interactions were: 
{action_history}

Available clinical tools you can call: 
{available_tools}

Continue with step {current_step}. First reason in <think> </think>, then output the next diagnostic or treatment action inside <action> </action>.
"""
