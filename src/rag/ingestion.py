import docx
import os
import json
import re

class DocumentIngestor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_document(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        doc = docx.Document(self.file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        return '\n'.join(full_text)

class TextChunker:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_by_article(self, text):
        pattern = r"(Article \d+\..*?)(?=Article \d+\.|$)"
        articles = re.findall(pattern, text, re.DOTALL)
        
        chunked_data = []
        for article in articles:
            match = re.search(r"Article (\d+)\.", article)
            article_num = match.group(1) if match else "?"
            
            chunked_data.append({
                "text": article.strip(),
                "metadata": {
                    "article_number": article_num,
                    "type": "article"
                }
            })
        return chunked_data

    def split_text(self, text):
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            
            if end < text_length:
                delimiter = text.rfind(' ', start, end)
                if delimiter != -1:
                    end = delimiter
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = max(start + 1, end - self.overlap)

        return chunks

class IngestionPipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.ingestor = DocumentIngestor(input_path)
        self.chunker = TextChunker()

    def run(self):
        print(f"Loading from {self.input_path}...")
        raw_text = self.ingestor.load_document()
        
        print("Chunking...")
        chunks = self.chunker.chunk_by_article(raw_text)
        
        if not chunks: 
            print("Regex split failed/empty, falling back to sliding window.")
            raw_chunks = self.chunker.split_text(raw_text)
            chunks = [{"text": c} for c in raw_chunks]

        print(f"Saving {len(chunks)} chunks to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print("Done.")

if __name__ == "__main__":
    from config import DATASET_PATH, CHUNKS_FILE_PATH
    pipeline = IngestionPipeline(DATASET_PATH, CHUNKS_FILE_PATH)
    pipeline.run()
