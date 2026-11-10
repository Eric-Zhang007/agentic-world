from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, save_json
from agent.client import AgentClient

config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
message = AgentClient(config["agent"]).complete([
    {"role": "user", "content": 'Return only JSON {"status":"ok"}.'}
])
save_json(Path(__file__).resolve().parents[1] / "outputs/agent_connectivity.json", message)
print(message)
