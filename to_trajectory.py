"""
Convert a raw Collab-Overcooked episode JSON (written by src/main.py) into a
canonical Trajectory (malrm.data.schema).

Episode JSON layout:
  {
    "total_timestamp": [0, 1, ...],
    "total_order_finished": ["boiled_egg"],
    "total_score": 20,
    "content": [
      {
        "timestamp": 0,
        "content": {
          "content": [
            [{"agent": 0|1, "analysis": "...", "say": "...", "plan": "..."}, ...],
            [{"agent": 0|1, "analysis": "...", "say": "...", "plan": "..."}, ...]
          ]
        }
      },
      ...
    ]
  }

Each game timestep contains a multi-turn LLM dialog (chef ↔ assistant).
Every dialog turn becomes one Step in the trajectory.
"""

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from malrm.data.schema import Step, Trajectory

_AGENT_NAMES = {0: "chef", 1: "assistant"}


def _turn_text(turn: dict) -> str:
    analysis = turn.get("analysis", "").strip()
    plan = str(turn.get("plan", "")).strip()
    say = turn.get("say", "").strip()
    return f"{analysis}\nAction: {plan}\nSay: {say}"


def episode_to_trajectory(episode_path: Path, model: str | None = None) -> Trajectory:
    """Load one episode JSON and return a Trajectory."""
    episode_path = Path(episode_path)
    with open(episode_path) as f:
        log = json.load(f)

    thread_id = episode_path.stem

    order_finished = log.get("total_order_finished", [])
    problem = order_finished[0] if order_finished else thread_id.rsplit("_", 1)[-1]

    if model is None:
        # path: .../data/{model}/{order}/experiment_*.json  →  parent.parent.name
        try:
            model = episode_path.parent.parent.name
        except Exception:
            model = "unknown"

    steps: list[Step] = []
    for entry in log.get("content", []):
        t = entry.get("timestamp", 0)
        sides = entry.get("content", {}).get("content", [[], []])
        for side in sides:
            for turn in side:
                agent_idx = turn.get("agent", 0)
                steps.append(Step(
                    agent=_AGENT_NAMES.get(agent_idx, f"agent{agent_idx}"),
                    t=t,
                    role="turn",
                    text=_turn_text(turn),
                ))

    success = log.get("total_score", 0) > 0

    meta = {
        "task_id": problem,
        "num_agents": 2,
        "step_count": len(log.get("total_timestamp", [])),
        "total_score": log.get("total_score", 0),
        "model": model,
        "orders_completed": log.get("total_order_finished", []),
    }

    return Trajectory(
        thread_id=thread_id,
        problem=problem,
        steps=steps,
        success=success,
        meta=meta,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert one episode JSON to Trajectory and print it")
    p.add_argument("episode_json", type=Path)
    args = p.parse_args()

    traj = episode_to_trajectory(args.episode_json)
    print(json.dumps(traj.to_dict(), indent=2))
