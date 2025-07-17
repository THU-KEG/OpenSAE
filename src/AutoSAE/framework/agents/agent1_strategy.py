import os
import json
from framework.agent_base import AgentBase
from openai import OpenAI

class Agent1Strategy(AgentBase):
    def __init__(self, settings: dict):
        super().__init__(settings)
        api_cfg = settings.get("api_client", {})
        self.client = OpenAI(
            api_key=api_cfg.get("api_key"),
            base_url=api_cfg.get("base_url")
        )

    def run(self, data: dict) -> dict:
        concept    = data.get("concept", "")
        definition = data.get("definition", "")
        prompt     = self.settings["prompt"].format(concept=concept, definition=definition)

        response = self.client.chat.completions.create(
            model=self.settings.get("model"),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.get("temperature", 1.0),
            max_tokens=self.settings.get("max_tokens", 512),
            top_p=self.settings.get("top_p", 1.0),
            n=1
        )

        content = response.choices[0].message.content.strip()
        try:
            payload = json.loads(content)
            tips = payload.get("tips", "")
        except json.JSONDecodeError:
            tips = content
        return {"tips": tips}