import json
import os
from framework.scheduler import Scheduler
from framework.agents.agent1_strategy import Agent1Strategy
from framework.agents.agent2_generate import Agent2Generate
from framework.agents.agent3_analyze import Agent3Analyze
from framework.agents.agent4_select import Agent4Select
from settings import SETTINGS

if __name__ == "__main__":
    concept    = "love"
    definition = "deep affection"
    layer_str = "00"

    # Agent1: tips
    s1 = Scheduler([Agent1Strategy], SETTINGS)
    out1 = s1.dispatch({"concept": concept, "definition": definition})
    tips = out1["tips"]
    print("Agent1 Tips:", tips)

    # Agent2: pairs
    s2 = Scheduler([Agent2Generate], SETTINGS)
    out2 = s2.dispatch({"concept": concept, "tips": tips})
    print("Agent2 Pairs:", json.dumps(out2["pairs"], indent=2, ensure_ascii=False))

    # Agent3: raw interpret
    cfg3 = SETTINGS["Agent3Analyze"]
    if "sae_ckpt_template" in cfg3:
        cfg3["sae_ckpt_path"] = cfg3.pop("sae_ckpt_template").format(layer_str=layer_str)
    s3 = Scheduler([Agent3Analyze], SETTINGS)
    out3 = s3.dispatch({"concept": concept})
    print("Agent3 Output:", json.dumps(out3, indent=2, ensure_ascii=False))

    # Prepare input for Agent4
    select_input = {
        "concept":          out3["concept"],
        "layer":            out3["layer"],
        "positive_sentences": out3.get("positive_sentences", []),
        "base_vectors":     out3["base_vectors"]
    }

    # Agent4: select and explain
    s4 = Scheduler([Agent4Select], SETTINGS)
    out4 = s4.dispatch(select_input)
    print("Agent4 Selection:", json.dumps(out4, indent=2, ensure_ascii=False))