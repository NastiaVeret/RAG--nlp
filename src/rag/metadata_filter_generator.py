"""
Metadata Filter Generator
Generates metadata filters based on user queries using LLM
"""

import os
import json
from typing import Dict, Optional, List, Set
from dotenv import load_dotenv
from groq import Groq
import config

load_dotenv()

CATEGORIES = ['Economy', 'Governance', 'Judiciary', 'Rights']

TOPICS = [
    'Constitution', 'Sovereignty', 'Human Rights', 'Citizenship', 'Democracy', 'Rule of Law',
    'Parliament', 'President', 'Cabinet of Ministers', 'Executive Power',
    'Judiciary', 'Constitutional Court', 'Justice', 'Law Enforcement',
    'Elections', 'Referendum', 'Local Self-Government',
    'National Security', 'Defense', 'State of Emergency', 'Martial Law', 
    'State Budget', 'Taxation', 'Economy', 'Private Property',
    'Freedom of Speech', 'Freedom of Assembly', 'Freedom of Religion',
    'Right to Privacy', 'Right to Education', 'Healthcare', 'Right to Labor',
    'Social Protection', 'Environment', 'Territorial Integrity', 'State Symbols',
    'National Bank', 'Territorial Structure'
]


class MetadataFilterGenerator:
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key is required")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        
        self.available_categories = CATEGORIES
        self.available_topics = TOPICS
    
    def generate_filter(self, query: str) -> Optional[Dict]:
        prompt = self._create_filter_prompt(query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a metadata filter generator. Analyze user queries and generate appropriate metadata filters in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            filter_text = response.choices[0].message.content.strip()
            filter_dict = self._parse_filter_response(filter_text)
            
            return filter_dict
            
        except Exception as e:
            print(f"Warning: Filter generation error: {e}")
            return None

    def _create_filter_prompt(self, query: str) -> str:
        categories_str = ", ".join(self.available_categories)
        topics_str = ", ".join(self.available_topics)

        prompt = f"""Analyze the following user query and generate metadata filters for document retrieval.

User Query: "{query}"

Available metadata fields:
1. category: {categories_str}
2. topics: {topics_str}
3. article_number: Any article number (e.g., "1", "24", "42")

Instructions:
- If the query mentions specific topics or categories, include them in the filter
- If the query mentions article numbers, include them
- If no specific metadata is mentioned, return an empty filter {{}}
- Return ONLY a valid JSON object, nothing else

Examples:

Query: "What are human rights in Ukraine?"
Filter: {{"category": "Rights", "topics": ["Human Rights"]}}

Query: "Tell me about Article 24"
Filter: {{"article_number": "24"}}

Query: "Constitution governance powers"
Filter: {{"category": "Governance", "topics": ["Constitution", "Executive Power"]}}

Query: "What is the capital of Ukraine?"
Filter: {{}}

Now generate the filter for the user query above. Return ONLY the JSON object:"""
        
        return prompt
    
    def _parse_filter_response(self, response: str) -> Optional[Dict]:
        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split('\n')
                if lines[0].startswith("```"):
                     lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                     lines = lines[:-1]
                response = '\n'.join(lines)
            
            filter_dict = json.loads(response)
            validated_filter = self._validate_filter(filter_dict)
            
            return validated_filter
            
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parse error: {e}")
            print(f"LLM response: {response}")
            return None
    
    def _validate_filter(self, filter_dict: Dict) -> Dict:
        validated = {}
        
        if "category" in filter_dict:
            category = filter_dict["category"]
            if category in self.available_categories:
                validated["category"] = category
        
        if "topics" in filter_dict:
            topics = filter_dict["topics"]
            if isinstance(topics, list):
                valid_topics = [t for t in topics if t in self.available_topics]
                if valid_topics:
                    validated["topics"] = valid_topics
            elif isinstance(topics, str) and topics in self.available_topics:
                validated["topics"] = [topics]
        
        if "article_number" in filter_dict:
            val = filter_dict["article_number"]
            if isinstance(val, (str, int)):
                validated["article_number"] = str(val)
        
        return validated
    
    def explain_filter(self, filter_dict: Dict) -> str:
        if not filter_dict:
            return "No metadata filters applied"
        
        explanations = []
        
        if "category" in filter_dict:
            explanations.append(f"Category: {filter_dict['category']}")
        
        if "topics" in filter_dict:
            topics = filter_dict["topics"]
            if isinstance(topics, list):
                explanations.append(f"Topics: {', '.join(topics)}")
            else:
                explanations.append(f"Topic: {topics}")
        
        if "article_number" in filter_dict:
            explanations.append(f"Article Number: {filter_dict['article_number']}")
        
        return "Filters: " + " | ".join(explanations)
