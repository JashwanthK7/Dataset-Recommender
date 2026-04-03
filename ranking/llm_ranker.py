import json
import os
import logging
import re
import ast
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompt import RANKING_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class LLMRanker:
    def __init__(self, model_id: str = 'Qwen/Qwen2.5-7B-Instruct'):
        self.model_id = model_id
        self.token = os.environ.get('HF_TOKEN', '')
        self.client = InferenceClient(model=self.model_id, token=self.token)

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {'role': 'system', 'content': 'You are a dataset evaluator. Return ONLY a valid array.'},
            {'role': 'user', 'content': prompt}
        ]
        
        @retry(
            stop=stop_after_attempt(3), 
            wait=wait_exponential(multiplier=2, min=5, max=20),
            reraise=True
        )
        def _request():
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=800,
                temperature=0.4 
            )
            return response.choices[0].message.content
            
        return _request()

    def rank(self, query: str, candidates: list[dict], intent_context: dict, active_constraints: dict) -> list[dict]:
        if not candidates:
            return []
            
        candidates_text = ''
        for i, c in enumerate(candidates):
            n = c.get('name', 'Unknown')
            s = c.get('source', 'Unknown')
            sz = c.get('size_estimate', 'Unknown')
            f = c.get('format', 'Unknown')
            d = c.get('description', 'No description provided.')
            candidates_text += f'Dataset {i+1}: {n}\nSource: {s}\nSize: {sz}\nFormat: {f}\nDescription: {d}\n\n'

        prompt = RANKING_PROMPT_TEMPLATE.format(
            query=query,
            intent_context=json.dumps(intent_context),
            active_constraints=json.dumps(active_constraints),
            candidates_text=candidates_text
        )

        try:
            raw_response = self._call_llm(prompt)
            
            match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if not match:
                raise ValueError('No array found in response')
            
            clean_text = match.group(0)
            
            llm_evaluations = ast.literal_eval(clean_text)
            eval_map = {str(item.get('name', '')).strip().lower(): item for item in llm_evaluations}
            
            for c in candidates:
                search_name = str(c.get('name', '')).strip().lower()
                eval_data = eval_map.get(search_name, {})
                
                raw_score = float(eval_data.get('llm_score', 5.0))
                normalized_score = raw_score / 10.0 if raw_score > 1.0 else raw_score
                
                c['llm_score'] = normalized_score
                c['suitability_notes'] = eval_data.get('reasoning', 'No detailed reasoning provided.')
                
        except Exception as e:
            logger.error(f'LLM Ranking failed: {e}')
            for c in candidates:
                c['llm_score'] = 0.5
                c['suitability_notes'] = 'LLM evaluation failed or timed out.'

        return candidates