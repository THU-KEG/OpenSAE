import os
import json
from framework.agent_base import AgentBase

class Agent4Select(AgentBase):
    """
    Reads raw interpret output (Agent3), original pairs, selects top 0-3 base vectors by FRC,
    and saves a distilled JSON with interpretations.
    """
    def __init__(self, settings: dict):
        super().__init__(settings)

    def run(self, data: dict) -> dict:
        # Extract state
        concept      = data.get("concept")
        layer        = data.get("layer")
        base_vectors = data.get("base_vectors", [])

        # Load original sentences
        # Determine project root based on this file's location
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pairs_fp = os.path.join(root, "data", "pairs", f"{concept}.json")
        try:
            pairs = json.load(open(pairs_fp, encoding="utf-8"))["pairs"]
            positives = [p.get("positive") for p in pairs]
        except Exception:
            positives = []

        # Select top 3 by frc
        sorted_bvs = sorted(base_vectors, key=lambda bv: bv.get("frc", 0.0), reverse=True)
        selected = sorted_bvs[:3]

        # Prepare result
        result = {
            "concept": concept,
            "layer": layer,
            "selected_base_vectors": []
        }
        for bv in selected:
            result["selected_base_vectors"].append({
                "base_vector_id": bv.get("base_vector_id"),
                "interpretation": bv.get("interpretation"),
                "ps": bv.get("ps"),
                "pn": bv.get("pn"),
                "frc": bv.get("frc")
            })

        # Validate JSON
        try:
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Result not JSON serializable: {e}")

        # Save to data/interpret/<concept>_<layer>.json
        out_dir = os.path.join(root, "data", "interpret")
        os.makedirs(out_dir, exist_ok=True)
        out_fp = os.path.join(out_dir, f"{concept}_{layer}.json")
        with open(out_fp, "w", encoding="utf-8") as f:
            f.write(json_str)

        return result