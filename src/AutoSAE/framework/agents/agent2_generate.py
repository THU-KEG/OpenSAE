import os
import json
from framework.agent_base import AgentBase
from openai import OpenAI

class Agent2Generate(AgentBase):
    """
    Agent to generate minimal contrast pairs and save output JSON directly.
    """
    def __init__(self, settings: dict):
        super().__init__(settings)
        api_cfg = settings.get("api_client", {})
        self.client = OpenAI(
            api_key=api_cfg.get("api_key"),
            base_url=api_cfg.get("base_url")
        )

    def run(self, data: dict) -> dict:
        # Prepare inputs
        concept = data.get("concept", "")
        tips    = data.get("tips", "")
        raw_prompt = self.settings.get("prompt", "")
        prompt = raw_prompt.format(concept=concept, tips=tips)

        # Call ChatCompletion
        response = self.client.chat.completions.create(
            model=self.settings.get("model"),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.get("temperature", 1.0),
            max_tokens=self.settings.get("max_tokens", 2048),
            top_p=self.settings.get("top_p", 1.0),
            n=1
        )

        # Extract content and strip code fences
        content = response.choices[0].message.content.strip()
        # Remove markdown fences if present
        if content.startswith("```") and content.endswith("```"):
            # remove leading ```[lang] and trailing ```
            lines = content.splitlines()
            # drop first and last lines
            lines = lines[1:-1]
            content = "\n".join(lines).strip()

        # Parse JSON directly
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from model output: {e}\nOutput was:\n{content}")

        # Save JSON to file
        base_dir = os.path.join(os.getcwd(), "AutoSAE", "data", "pairs")
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, f"{concept}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload