import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMService:
    def __init__(self, api_key=None):
        """Initialize LLM service with Groq API"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key is required")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def generate_response(self, query, context_chunks):
        """
        Synthesizes an answer based on the query and retrieved context chunks.
        Uses LLM to generate a coherent answer from the context.
        
        Args:
            query: User's question
            context_chunks: List of retrieved document chunks
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        if not context_chunks:
            return {
                "answer": "I couldn't find any specific information answering that question.",
                "sources": []
            }

        # Extract text from chunks
        sources_text = [
            chunk.get('text', str(chunk)) 
            for chunk in context_chunks
        ]
        
        # Create context for LLM
        context = "\n\n".join([
            f"[Source {i+1}]\n{text}" 
            for i, text in enumerate(sources_text)
        ])
        
        # Create prompt for LLM
        prompt = self._create_prompt(query, context)
        
        try:
            # Generate answer using LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always cite sources using [1], [2], etc."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            generated_answer = response.choices[0].message.content.strip()
            
            return {
                "answer": generated_answer,
                "sources": sources_text
            }
            
        except Exception as e:
            print(f"Warning: LLM generation error: {e}")
            # Fallback to simple extraction
            return self._fallback_response(query, sources_text)
    
    def _create_prompt(self, query, context):
        """Create a prompt for the LLM"""
        
        prompt = f"""Based on the following context from the Ukrainian Constitution, answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Provide a clear, concise answer based ONLY on the information in the context
- Cite sources using [1], [2], [3] etc. corresponding to the source numbers
- If the context doesn't contain enough information, say so
- Keep the answer focused and relevant to the question
- Use proper grammar and complete sentences

Answer:"""
        
        return prompt
    
    def _fallback_response(self, query, sources_text):
        """Fallback response if LLM fails"""
        import re
        
        if not sources_text:
            return {
                "answer": "I couldn't find any specific information answering that question.",
                "sources": []
            }
        
        top_text = sources_text[0]
        clean_answer = re.sub(r'^Article \d+\.\s*', '', top_text)
        generated_answer = f"{clean_answer} [1]"
        
        return {
            "answer": generated_answer,
            "sources": sources_text
        }
