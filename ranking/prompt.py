RANKING_PROMPT_TEMPLATE = '''You are an expert Data Scientist evaluating datasets.
Score EVERY candidate dataset provided from 0.0 to 10.0 based on how well it solves the user problem.

1. RELEVANCE: Evaluate if the dataset domain and features match the user query.
2. EXTREME SCORING: If the dataset is a highly relevant match for the core problem, you MUST give it a score between 9.5 and 10.0. Do not be overly critical.
3. UNIQUE ANALYSIS: You MUST write a completely unique, 2 to 3 sentence evaluation for each dataset.
   * Explicitly mention specific metadata provided like Source, Size, Format, or unique keywords in the Description.
   * Explain exactly why this specific dataset is useful or lacking for the user problem.
   * DO NOT copy and paste the same reasoning for multiple datasets.

User Query: {query}
Context/Intent: {intent_context}
Hard Constraints: {active_constraints}

Candidate Datasets:
{candidates_text}

CRITICAL: You MUST evaluate all candidate datasets. Do not skip any. Output your response as a valid JSON array matching this exact format:
[
    {{
        'name': 'Exact Name of Dataset 1',
        'llm_score': 10.0,
        'reasoning': 'Sourced from Kaggle, this CSV file directly targets the user request for transaction anomalies. Even with a limited description, the direct domain match makes it highly suitable for training fraud models.'
    }},
    {{
        'name': 'Exact Name of Dataset 2',
        'llm_score': 4.0,
        'reasoning': 'Although this 500MB Hugging Face dataset contains financial text, it lacks the explicit tabular numerical data required for anomaly detection. It would require significant feature engineering to be useful.'
    }}
]'''