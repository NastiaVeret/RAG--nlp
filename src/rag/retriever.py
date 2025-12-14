import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import config

class Retriever:
    def __init__(self, chunks_file=config.CHUNKS_FILE_PATH):
        self.chunks_file = chunks_file
        self.chunks = self._load_chunks()
        self.encoder = self._load_model()
        self.index = self._build_index()

    def _load_chunks(self):
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_model(self):
        return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def _build_index(self):
        corpus_texts = [self._get_text(doc) for doc in self.chunks]
        embeddings = self.encoder.encode(corpus_texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def _get_text(self, chunk):
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        return str(chunk)

    def _matches_filter(self, chunk: Dict, metadata_filter: Dict) -> bool:
        if not metadata_filter:
            return True
        
        chunk_metadata = chunk.get('metadata', {})
        
        if 'category' in metadata_filter:
            if chunk_metadata.get('category') != metadata_filter['category']:
                return False
        
        if 'article_number' in metadata_filter:
            if chunk_metadata.get('article_number') != metadata_filter['article_number']:
                return False
        
        if 'topics' in metadata_filter:
            filter_topics = metadata_filter['topics']
            chunk_topics = chunk_metadata.get('topics', [])
            
            if isinstance(filter_topics, list):
                if not any(topic in chunk_topics for topic in filter_topics):
                    return False
            elif isinstance(filter_topics, str):
                if filter_topics not in chunk_topics:
                    return False
        
        return True

    def search_semantic(self, query, top_k=3, metadata_filter: Optional[Dict] = None):
        search_k = top_k * 5 if metadata_filter else top_k
        
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector, search_k)
        
        results = []
        for i in range(search_k):
            idx = indices[0][i]
            dist = distances[0][i]
            chunk = self.chunks[idx]
            
            if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                continue
            
            results.append({
                "chunk": chunk,
                "score": float(dist),
                "id": int(idx)
            })
            
            if len(results) >= top_k:
                break
        
        return results
