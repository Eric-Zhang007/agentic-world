import os

import requests


class AgentClient:
    def __init__(self, config):
        self.config = config

    def complete(self, messages, tools=None):
        body = {"model": self.config["model"], "messages": messages}
        if tools:
            body["tools"] = tools
        response = requests.post(
            self.config["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + os.getenv("OPENAI_API_KEY", self.config.get("api_key", ""))},
            json=body, timeout=self.config.get("timeout", 180),
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("choices"):
            raise RuntimeError("Gateway response has no choices")
        return data["choices"][0]["message"]
