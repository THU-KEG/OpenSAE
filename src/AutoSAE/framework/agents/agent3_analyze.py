import os
import re
import json
import torch
import transformers
from framework.agent_base import AgentBase
from openai import OpenAI
from opensae import OpenSae
from opensae.transformer_with_sae import TransformerWithSae

# Utility to extract layer index

def extract_layer_index_from_path(path: str) -> str:
    m = re.search(r'Layer_(\d+)', path)
    return m.group(1) if m else "00"

# Compute FRC stats
def compute_frc(structured_data: list, top_k: int = 10):
    pos = [s for i, s in enumerate(structured_data) if i % 2 == 0]
    neg = [s for i, s in enumerate(structured_data) if i % 2 == 1]
    num_pos, num_neg = len(pos), len(neg)
    all_bases = set(act["base_vector"] for s in structured_data for t in s["tokens"] for act in t["activations"])
    frc_list, stats_map = [], {}
    for b in all_bases:
        ps = sum(any(act["base_vector"] == b for t in s["tokens"] for act in t["activations"]) for s in pos) / max(num_pos,1)
        pn = sum(not any(act["base_vector"] == b for t in s["tokens"] for act in t["activations"]) for s in neg) / max(num_neg,1)
        frc = (2 * ps * pn / (ps + pn)) if (ps + pn) > 0 else 0.0
        frc_list.append((b, ps, pn, frc))
        stats_map[b] = {"ps": ps, "pn": pn, "frc": frc}
    frc_list.sort(key=lambda x: x[3], reverse=True)
    top_bases = [b for b, _, _, _ in frc_list[:top_k]]
    return frc_list, top_bases, stats_map

class Agent3Analyze(AgentBase):
    """
    Combines FRC computation and semantic analysis:
      - computes top-K FRC base vectors
      - prepares activation data
      - calls LLM to interpret each vector
      - saves raw interpret JSON per concept and layer
    """
    def __init__(self, settings: dict):
        super().__init__(settings)
        api_cfg = settings.get("api_client", {})
        self.client = OpenAI(
            api_key=api_cfg.get("api_key"),
            base_url=api_cfg.get("base_url")
        )
        self.sae_ckpt = settings.get("sae_ckpt_path")
        self.lm_path = settings.get("lm_model_path")
        self.device = settings.get("device", "cuda:0")
        self.top_k = settings.get("top_k", 10)
        self.prompt = settings.get("analysis_prompt")
        self.analysis_model = settings.get("analysis_model")
        self.analysis_temp = settings.get("analysis_temp", 0.01)
        self.analysis_max_tokens = settings.get("analysis_max_tokens", 512)

    def run(self, data: dict) -> dict:
        concept = data.get("concept")
        # Load minimal contrast pairs
        pairs_fp = os.path.join(os.getcwd(), "AutoSAE", "data", "pairs", f"{concept}.json")
        pairs = json.load(open(pairs_fp, encoding="utf-8"))["pairs"]
        sentences = [p["positive"] for p in pairs] + [p["counterfactual"] for p in pairs]

        # Load SAE and LLM
        layer_idx = extract_layer_index_from_path(self.sae_ckpt)
        sae = OpenSae.from_pretrained(self.sae_ckpt)
        lm = TransformerWithSae(self.lm_path, sae, self.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.lm_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        enc = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        structured = lm.extract_data(enc, tokenizer)

        # Compute FRC
        _, top_bases, stats_map = compute_frc(structured, top_k=self.top_k)
        # Prepare analysis input
        base_vectors = []
        for b in top_bases:
            tokens, activations = [], []
            for sent in structured:
                for tok in sent["tokens"]:
                    for act in tok["activations"]:
                        if act["base_vector"] == b:
                            tokens.append(tok["token"])
                            activations.append(act["activation"])
            stats = stats_map[b]
            base_vectors.append({
                "base_vector_id": b,
                "tokens": tokens,
                "activations": activations,
                **stats
            })
        analysis_input = {"layer": layer_idx, "base_vectors": base_vectors, "target_features": [concept]}

        # Call LLM
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": json.dumps({"analysis_input": analysis_input}, ensure_ascii=False)}
        ]
        resp = self.client.chat.completions.create(
            model=self.analysis_model,
            messages=messages,
            temperature=self.analysis_temp,
            max_tokens=self.analysis_max_tokens
        )
        content = resp.choices[0].message.content.strip()
        # Remove markdown fences
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]).strip()

        # Parse JSON
        try:
            output = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from Agent3Analyze:\n{content}")

        # Augment stats
        for bv in output.get("base_vectors", []):
            bid = bv.get("base_vector_id")
            if bid in stats_map:
                bv.update(stats_map[bid])
        output["layer"] = layer_idx

        # Save raw interpret JSON (concept_layer.json)
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        raw_dir = os.path.join(root, "data", "raw_interpret")
        os.makedirs(raw_dir, exist_ok=True)
        raw_fp = os.path.join(raw_dir, f"{concept}_{layer_idx}.json")
        with open(raw_fp, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return output