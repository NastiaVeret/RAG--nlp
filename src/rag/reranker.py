from sentence_transformers import CrossEncoder
import config

class ReRanker:
    def __init__(self, model_name=config.RERANKER_MODEL_NAME):
        try:
            self.model = CrossEncoder(model_name)
            self.enabled = True
        except Exception as e:
            print(f"Warning: Could not load ReRanker model ({e}). Re-ranking will be skipped.")
            self.enabled = False

    def rerank(self, query, initial_results, top_k=3):
        """
        Re-ranks a list of retrieved results using a Cross-Encoder.
        initial_results: list of dicts with 'chunk' and 'score'
        """
        if not self.enabled or not initial_results:
            return initial_results[:top_k]

        chunk_texts = [
            res['chunk'].get('text', str(res['chunk'])) 
            for res in initial_results
        ]
        pairs = [[query, text] for text in chunk_texts]

        scores = self.model.predict(pairs)

        for i, res in enumerate(initial_results):
            res['cross_score'] = float(scores[i])
        
        initial_results.sort(key=lambda x: x['cross_score'], reverse=True)

        return initial_results[:top_k]
