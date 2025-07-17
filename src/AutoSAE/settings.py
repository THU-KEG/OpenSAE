SETTINGS = {
    "Agent1Strategy": {
        "api_client": {"api_key": "<your_api_key>", "base_url": "<your_base_url>"},
        "prompt": '''
You are given:
  • concept: {concept}
  • definition: {definition}

Your task:
Provide a detailed analysis of how to generate minimal contrast pairs specifically for the given concept. Focus on:
  1. Identifying the core token or phrase of the concept in context.
  2. Choosing sentence templates or contexts that naturally accommodate the concept.
  3. Determining which minimal edit (deletion or substitution) would yield a valid counterfactual.
  4. Ensuring semantic and syntactic coherence after the edit.

Format:
Return only a JSON object with one field 'tips', whose value is your concept-specific analysis.
''',
        "model": "gpt-4o",
        "temperature": 0.01,
        "max_tokens": 500,
        "top_p": 1.0
    },

    "Agent2Generate": {
        "api_client": {"api_key": "<your_api_key>", "base_url": "<your_base_url>"},
        "prompt": '''
Use the following guidelines to generate minimal contrast pairs exactly as specified:
  - A minimal contrast pair consists of two sentences identical in every respect except for exactly one minimal edit to the target concept mention.
  - Allowed edits: deletion of the concept token or substitution with a synonym or more general term.
  - All other words, punctuation, and structure must remain unchanged.
  - Ensure semantic and syntactic coherence.
  - Aim for topical and syntactic diversity across examples.

You are given:
  • concept: {concept}
  • tips: {tips}

Your task (follow the above guidelines and 'tips' exactly):
1. Generate 10 positive sentences that each include the target '{concept}', are fluent, and contextually coherent.
2. For each positive sentence, produce a minimal-contrast counterfactual by applying only one minimal edit to the mention of '{concept}', removing its explicit reference to the concept while preserving grammatical structure.
3. Output a single JSON object with one field 'pairs', whose value is an array of 10 objects. Each object must have exactly two keys:
   - 'positive': the original sentence.
   - 'counterfactual': its minimally edited counterpart.

Return only the JSON result, without any additional text. Return ONLY the JSON object described, without code fences or extra text.

Examples:
Concept: 'apple' -> {{"pairs":[{{"positive":"She picked a ripe apple from the tree.","counterfactual":"She picked a ripe banana from the tree."}}]}}
Concept: 'England' -> {{"pairs":[{{"positive":"Tourists visited England in summer.","counterfactual":"Tourists visited the country in summer."}}]}}
''',
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 2000,
        "num_pairs": 10,
        "max_lines": 20,
        "top_p": 1.0
    },

    "Agent3Analyze": {
        "api_client": {"api_key": "<your_api_key>", "base_url": "<your_base_url>"},
        "sae_ckpt_template": "<your_sae_ckpt_template>", 
        "lm_model_path": "<your_lm_model_path>",                           
        "device": "cuda:0",
        "top_k": 10,
        "analysis_prompt": '''
You are an expert assistant for interpreting sparse autoencoder base vectors.
You will receive exactly one JSON object as input with this structure:

{
  "analysis_input": {
    "layer": "<two-digit layer string>",
    "base_vectors": [
      {
        "base_vector_id": <int>,
        "tokens": ["..."],
        "activations": [<float>, ...]
      }
      // up to top_k entries
    ],
    "target_features": ["{concept}"]
  }
}

Your task is to analyze how each base vector relates to the target concept. For each base_vector:
  1. Describe what aspect or sub-concept of "{concept}" it captures (e.g., thematic element, syntactic pattern, semantic nuance).
  2. Explain possible application scenarios or examples where this vector would strongly activate for the concept.
  3. Relate it back to how it helps represent the overall concept in the model.

Return a JSON object with:
- "layer": same layer string
- "base_vectors": list of objects each containing:
    - base_vector_id (int)
    - interpretation (string): concise description of what this base vector captures
    - ps, pn, frc (floats): the computed statistics
- "target_features": echoed input features

Do not include any additional commentary outside the JSON.
''',
        "analysis_model": "gpt-4o",
        "analysis_temp": 0.01,
        "analysis_max_tokens": 2048
    },

    "Agent4Select": {
        "prompt": '''
You are given:
  • concept: {concept}
  • layer: {layer}
  • positive_sentences: a list of original sentences illustrating the concept
  • base_vectors: a list of objects each with:
    - base_vector_id (int)
    - interpretation (string)
    - ps, pn, frc (floats)

Your task:
1. From the provided base_vectors, select up to 3 that have the highest frc values.
2. For each selected base vector, explain:
   - Why this vector is most representative of "{concept}".
   - What sub-aspect or nuance of the concept it captures.
   - How it could be applied or detected in real text.

Output:
Return a JSON object with:
- "concept": same concept
- "layer": same layer
- "selected_base_vectors": an array of up to 3 objects each containing:
    - base_vector_id
    - interpretation
    - ps, pn, frc
    - explanation (string)

Format:
Return only the JSON object described, without any additional text.
''',
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 512
    }
}