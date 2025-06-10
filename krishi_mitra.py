import gradio as gr
import json
import os
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
import requests
from groq import Groq
import re
import hashlib
import chromadb
from chromadb.config import Settings
import uuid
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.llms.base import LLM
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    FERTILIZER_ADVICE = "fertilizer_advice"
    CROP_SELECTION = "crop_selection"
    IRRIGATION_GUIDANCE = "irrigation_guidance"
    PEST_MANAGEMENT = "pest_management"
    SOIL_MANAGEMENT = "soil_management"
    DISEASE_DIAGNOSIS = "disease_diagnosis"
    WEATHER_RELATED = "weather_related"
    MARKET_PRICES = "market_prices"
    GENERAL_FARMING = "general_farming"
    URGENT_ISSUE = "urgent_issue"
    GREETING = "greeting"
    


class QueryClassifier:
    def __init__(self):
        # Define keyword patterns for different query types
        self.classification_patterns = {
            QueryType.FERTILIZER_ADVICE: {
                'keywords': [
                    'fertilizer', 'खाद', 'उर्वरक', 'npk', 'urea', 'डाप', 'dap', 
                    'compost', 'manure', 'organic', 'chemical', 'nutrient', 'पोषक'
                ],
                'priority': 2
            },
            QueryType.PEST_MANAGEMENT: {
                'keywords': [
                    'pest', 'कीट', 'insect', 'bug', 'termite', 'aphid', 'bollworm',
                    'pesticide', 'spray', 'neem', 'कीटनाशक', 'दीमक', 'माहू'
                ],
                'priority': 3  # High priority for pest issues
            },
            QueryType.DISEASE_DIAGNOSIS: {
                'keywords': [
                    'disease', 'बीमारी', 'fungus', 'virus', 'bacteria', 'leaf spot',
                    'wilt', 'rot', 'blight', 'मुरझाना', 'पत्ती', 'दाग', 'सड़न'
                ],
                'priority': 3  # High priority for diseases
            },
            QueryType.IRRIGATION_GUIDANCE: {
                'keywords': [
                    'water', 'पानी', 'irrigation', 'सिंचाई', 'drip', 'sprinkler',
                    'flood', 'drought', 'watering', 'moisture', 'बूंद-बूंद'
                ],
                'priority': 2
            },
            QueryType.CROP_SELECTION: {
                'keywords': [
                    'crop', 'फसल', 'seed', 'बीज', 'variety', 'किस्म', 'selection',
                    'choose', 'best', 'suitable', 'चुनना', 'उपयुक्त'
                ],
                'priority': 1
            },
            QueryType.SOIL_MANAGEMENT: {
                'keywords': [
                    'soil', 'मिट्टी', 'ph', 'testing', 'जांच', 'fertility', 'उर्वरता',
                    'organic matter', 'tillage', 'जुताई', 'खाद', 'compaction'
                ],
                'priority': 2
            },
            QueryType.WEATHER_RELATED: {
                'keywords': [
                    'weather', 'मौसम', 'rain', 'बारिश', 'temperature', 'तापमान',
                    'season', 'मौसम', 'climate', 'जलवायु', 'frost', 'पाला'
                ],
                'priority': 2
            },
            QueryType.MARKET_PRICES: {
                'keywords': [
                    'price', 'दाम', 'market', 'मंडी', 'sell', 'बेचना', 'cost',
                    'profit', 'मुनाफा', 'rate', 'भाव', 'mandi'
                ],
                'priority': 1
            },
            QueryType.URGENT_ISSUE: {
                'keywords': [
                    'urgent', 'तुरंत', 'emergency', 'आपातकाल', 'dying', 'मर रहा',
                    'help', 'मदद', 'immediate', 'तत्काल', 'crisis', 'संकट'
                ],
                'priority': 4  # Highest priority
            },
            QueryType.GREETING: {
                'keywords': [
                    'hello', 'hi', 'नमस्ते', 'हैलो', 'good morning', 'good evening',
                    'how are you', 'kaise ho', 'आप कैसे हैं'
                ],
                'priority': 0
            }
        }
    
    def classify_query(self, query: str, user_context: Dict = None) -> Tuple[QueryType, float, Dict]:
        """Classify the user query and return type, confidence, and analysis"""
        query_lower = query.lower()
        
        # Score each category
        category_scores = {}
        matched_keywords = {}
        
        for query_type, pattern_info in self.classification_patterns.items():
            score = 0
            keywords_found = []
            
            for keyword in pattern_info['keywords']:
                if keyword in query_lower:
                    score += 1
                    keywords_found.append(keyword)
            
            # Apply priority weighting
            weighted_score = score * pattern_info['priority'] if score > 0 else 0
            
            category_scores[query_type] = weighted_score
            matched_keywords[query_type] = keywords_found
        
        # Find the best match
        if not any(category_scores.values()):
            best_category = QueryType.GENERAL_FARMING
            confidence = 0.5
        else:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            max_score = max(category_scores.values())
            total_possible = max_score + 1  # Avoid division by zero
            confidence = max_score / total_possible

        # Analysis for debugging and insights
        analysis = {
            'all_scores': {qt.value: score for qt, score in category_scores.items()},
            'matched_keywords': matched_keywords.get(best_category, []),
            'query_length': len(query.split()),
            'user_crops': user_context.get('crops', '') if user_context else '',
            'user_location': user_context.get('location', '') if user_context else ''
        }
        return best_category, confidence, analysis
class SpecializedResponseGenerator:
    def __init__(self, groq_client):
        self.groq_client = groq_client
        
    def generate_fertilizer_advice(self, query: str, user_context: Dict, language: str) -> str:
        """Generate specialized fertilizer advice"""
        system_prompt = f"""You are a fertilizer specialist for Indian farmers. 
        
        User Profile: Location: {user_context.get('location', 'Not specified')}, 
        Crops: {user_context.get('crops', 'Not specified')}
        
        Focus on:
        1. NPK ratios specific to their crops and soil
        2. Application timing and methods
        3. Organic vs chemical options
        4. Cost-effective solutions
        5. Local availability in India
        
        Language: {"Hindi primarily" if language == "Hindi" else "English primarily" if language == "English" else "Mix Hindi and English"}
        
        Be specific, practical, and consider Indian farming conditions."""
        
        return self._get_specialized_response(system_prompt, query)
    
    def generate_pest_management(self, query: str, user_context: Dict, language: str) -> str:
        """Generate specialized pest management advice"""
        system_prompt = f"""You are a pest management expert for Indian agriculture.
        
        User Profile: Location: {user_context.get('location', 'Not specified')}, 
        Crops: {user_context.get('crops', 'Not specified')}
        
        Focus on:
        1. Identify the pest based on description
        2. Integrated Pest Management (IPM) approach
        3. Biological control methods first
        4. Chemical pesticides as last resort
        5. Prevention strategies
        6. Safe application practices
        
        Language: {"Hindi primarily" if language == "Hindi" else "English primarily" if language == "English" else "Mix Hindi and English"}
        
        Prioritize farmer and environmental safety."""
        
        return self._get_specialized_response(system_prompt, query)
    
    def generate_irrigation_guidance(self, query: str, user_context: Dict, language: str) -> str:
        """Generate specialized irrigation advice"""
        system_prompt = f"""You are an irrigation specialist for Indian farmers.
        
        User Profile: Location: {user_context.get('location', 'Not specified')}, 
        Crops: {user_context.get('crops', 'Not specified')}
        
        Focus on:
        1. Water requirements for specific crops
        2. Efficient irrigation methods (drip, sprinkler)
        3. Water scheduling based on crop stage
        4. Water conservation techniques
        5. Monsoon and drought management
        6. Cost-effective solutions
        
        Language: {"Hindi primarily" if language == "Hindi" else "English primarily" if language == "English" else "Mix Hindi and English"}
        
        Consider water scarcity and Indian climate conditions."""
        
        return self._get_specialized_response(system_prompt, query)
    
    def generate_crop_selection(self, query: str, user_context: Dict, language: str) -> str:
        """Generate specialized crop selection advice"""
        system_prompt = f"""You are a crop consultant for Indian farmers.
        
        User Profile: Location: {user_context.get('location', 'Not specified')}, 
        Farm Size: {user_context.get('farm_size', 'Not specified')}
        
        Focus on:
        1. Climate suitability for the region
        2. Soil type requirements
        3. Market demand and profitability
        4. Water requirements
        5. Crop rotation benefits
        6. Local seed availability
        
        Language: {"Hindi primarily" if language == "Hindi" else "English primarily" if language == "English" else "Mix Hindi and English"}
        
        Recommend practical, profitable crops for Indian conditions."""
        
        return self._get_specialized_response(system_prompt, query)
    
    def generate_urgent_response(self, query: str, user_context: Dict, language: str) -> str:
        """Generate urgent/emergency farming advice"""
        system_prompt = f"""🚨 URGENT FARMING ASSISTANCE 🚨
        
        You are responding to an urgent farming issue. 
        
        User Profile: Location: {user_context.get('location', 'Not specified')}, 
        Crops: {user_context.get('crops', 'Not specified')}
        
        Provide:
        1. IMMEDIATE action steps (numbered list)
        2. What to do in the next 24 hours
        3. Warning signs to watch for
        4. When to contact local agricultural officer
        5. Emergency contact suggestions
        
        Language: {"Hindi primarily" if language == "Hindi" else "English primarily" if language == "English" else "Mix Hindi and English"}
        
        Be direct, actionable, and reassuring. Start with "🚨 URGENT ADVICE:"."""
        
        return self._get_specialized_response(system_prompt, query)
    
    def generate_general_response(self, query: str, user_context: Dict, language: str, relevant_contexts: List[str]) -> str:
        """Generate general farming advice"""
        context_text = "\n".join([f"- {ctx[:200]}" for ctx in relevant_contexts[:3]]) if relevant_contexts else ""
        
        system_prompt = f"""You are Krishi Mitra, a comprehensive farming assistant.
        
        User Profile: Location: {user_context.get('location', 'Not specified')}, 
        Crops: {user_context.get('crops', 'Not specified')}, 
        Farm Size: {user_context.get('farm_size', 'Not specified')}
        
        Relevant Knowledge:
        {context_text}
        
        Provide helpful, practical farming advice considering Indian agricultural practices.
        
        Language: {"Hindi primarily" if language == "Hindi" else "English primarily" if language == "English" else "Mix Hindi and English"}"""
        
        return self._get_specialized_response(system_prompt, query)
    
    def _get_specialized_response(self, system_prompt: str, query: str) -> str:
        """Get response from Groq with specialized system prompt"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in specialized response: {e}")
            return f"Error generating response: {str(e)}. Please try again."

# Initialize Groq client - using environment variable for Modal
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Changed to use environment variable
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

client = Groq(api_key=GROQ_API_KEY)

# Custom LangChain LLM wrapper for Groq
class GroqLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
                stop=stop
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "groq"

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Create data directory in current working directory if doesn't exist
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "krishi_mitra.db")
        else:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                session_id TEXT PRIMARY KEY,
                location TEXT,
                crops TEXT,
                farm_size TEXT,
                created_at TIMESTAMP,
                last_active TIMESTAMP
            )
        ''')
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_message TEXT,
                ai_response TEXT,
                language TEXT,
                timestamp TIMESTAMP,
                message_hash TEXT,
                FOREIGN KEY (session_id) REFERENCES users (session_id)
            )
        ''')
        
        # Farming knowledge base table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS farming_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                content TEXT,
                language TEXT,
                keywords TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_user_profile(self, session_id: str, location: str, crops: str, farm_size: str):
        """Save or update user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (session_id, location, crops, farm_size, created_at, last_active)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, location, crops, farm_size, 
              datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, session_id: str) -> Dict:
        """Get user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE session_id = ?', (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'session_id': result[0],
                'location': result[1] or '',
                'crops': result[2] or '',
                'farm_size': result[3] or '',
                'created_at': result[4],
                'last_active': result[5]
            }
        return {}
    
    def save_chat_message(self, session_id: str, user_message: str, ai_response: str, language: str):
        """Save chat message to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        message_hash = hashlib.md5(f"{user_message}{ai_response}".encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO chat_history 
            (session_id, user_message, ai_response, language, timestamp, message_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, user_message, ai_response, language, 
              datetime.now().isoformat(), message_hash))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message, ai_response, language, timestamp 
            FROM chat_history 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'user_message': r[0],
            'ai_response': r[1],
            'language': r[2],
            'timestamp': r[3]  
        } for r in reversed(results)]  # Reverse to get chronological order

class VectorMemoryManager:
    # def __init__(self, persist_directory: str = None):
    #     """Initialize ChromaDB for vector-based memory"""
    #     if persist_directory is None:
    #         self.persist_directory = os.path.join(os.getcwd(), "chroma_db")
    #     else:
    #         self.persist_directory = persist_directory
            
    #     os.makedirs(self.persist_directory, exist_ok=True)
        
    #     # Initialize ChromaDB
    #     self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
    #     # Create collections
    #     self.chat_collection = self.chroma_client.get_or_create_collection(
    #         name="chat_memory",
    #         metadata={"description": "Chat conversation memory"}
    #     )
        
    #     self.knowledge_collection = self.chroma_client.get_or_create_collection(
    #         name="farming_knowledge",
    #         metadata={"description": "Farming knowledge base"}
    #     )
        
    #     # Initialize farming knowledge base
    #     self.initialize_knowledge_base()


    def __init__(self, persist_directory: str = None):
        """Initialize ChromaDB for vector-based memory"""
        if persist_directory is None:
            self.persist_directory = os.path.join(os.getcwd(), "chroma_db")
        else:
            self.persist_directory = persist_directory
            
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB - FIX: use self.persist_directory
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Rest of the code remains the same...
    
    def initialize_knowledge_base(self):
        """Initialize basic farming knowledge base"""
        farming_knowledge = [
            {
                "id": "fertilizer_wheat",
                "content": "For wheat crop, use NPK fertilizer with ratio 120:60:40 kg per hectare. Apply in 3 splits - 50% nitrogen at sowing, 25% at first irrigation, 25% at second irrigation.",
                "keywords": ["wheat", "fertilizer", "NPK", "nitrogen", "गेहूं", "खाद"],
                "topic": "fertilizer_management"
            },
            {
                "id": "pest_cotton",
                "content": "Common cotton pests include bollworm, aphids, and whitefly. Use integrated pest management - spray neem oil early morning, install pheromone traps, and use biological pesticides.",
                "keywords": ["cotton", "pest", "bollworm", "aphids", "कपास", "कीट"],
                "topic": "pest_management"
            },
            {
                "id": "irrigation_rice",
                "content": "Rice requires 1200-1500mm water per season. Maintain 2-5cm standing water during vegetative growth. Reduce water 15 days before harvest.",
                "keywords": ["rice", "irrigation", "water", "धान", "पानी", "सिंचाई"],
                "topic": "irrigation"
            },
            {
                "id": "soil_testing",
                "content": "Test soil pH, nitrogen, phosphorus, potassium, and organic matter every 2-3 years. Ideal pH for most crops is 6.0-7.5. Collect samples from multiple points.",
                "keywords": ["soil", "testing", "pH", "nitrogen", "मिट्टी", "जांच"],
                "topic": "soil_management"
            }
        ]
        
        # Check if knowledge base is already populated
        try:
            existing_docs = self.knowledge_collection.count()
            if existing_docs == 0:
                # Add knowledge to vector store
                for knowledge in farming_knowledge:
                    self.knowledge_collection.add(
                        documents=[knowledge["content"]],
                        metadatas=[{
                            "topic": knowledge["topic"],
                            "keywords": ",".join(knowledge["keywords"])
                        }],
                        ids=[knowledge["id"]]
                    )
        except Exception as e:
            print(f"Error initializing knowledge base: {e}")
    
    def add_chat_memory(self, session_id: str, user_message: str, ai_response: str, language: str):
        """Add chat exchange to vector memory"""
        try:
            conversation_text = f"User: {user_message}\nAssistant: {ai_response}"
            
            self.chat_collection.add(
                documents=[conversation_text],
                metadatas=[{
                    "session_id": session_id,
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[f"{session_id}_{datetime.now().timestamp()}"]
            )
        except Exception as e:
            print(f"Error adding chat memory: {e}")
    
    def search_relevant_context(self, query: str, session_id: str, n_results: int = 3) -> List[str]:
        """Search for relevant context from chat history and knowledge base"""
        relevant_contexts = []
        
        try:
            # Search in chat history
            chat_results = self.chat_collection.query(
                query_texts=[query],
                where={"session_id": session_id},
                n_results=min(n_results, 2)
            )
            
            if chat_results['documents'] and chat_results['documents'][0]:
                relevant_contexts.extend(chat_results['documents'][0])
            
            # Search in knowledge base
            knowledge_results = self.knowledge_collection.query(
                query_texts=[query],
                n_results=min(n_results, 3)
            )
            
            if knowledge_results['documents'] and knowledge_results['documents'][0]:
                relevant_contexts.extend(knowledge_results['documents'][0])
                
        except Exception as e:
            print(f"Error searching context: {e}")
        
        return relevant_contexts[:n_results]

class KrishiMitra:
    def __init__(self):
        self.user_sessions = {}
        self.supported_languages = ["Hindi", "English", "HindEnglish"]
        
        # Initialize database and vector memory
        self.db_manager = DatabaseManager()
        self.vector_memory = VectorMemoryManager()
        
         # Initialize query classifier and response generator
        self.query_classifier = QueryClassifier()
        self.response_generator = SpecializedResponseGenerator(client)
        
        # Initialize LangChain memory for conversation summarization
        self.groq_llm = GroqLLM()
        self.conversation_memories = {}  # Session-based memory storage
        
    def detect_language(self, text: str) -> str:
        """Simple language detection based on script"""
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return "English"  # Default to English if empty message
        
        hindi_ratio = hindi_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if hindi_ratio > 0.7:
            return "Hindi"
        elif english_ratio > 0.7:
            return "English"
        else:
            return "HindEnglish"  # Mixed language
    def get_conversation_memory(self, session_id: str):
        """Get or create conversation memory for a session"""
        if session_id not in self.conversation_memories:
            self.conversation_memories[session_id] = ConversationSummaryBufferMemory(
                llm=self.groq_llm,
                max_token_limit=1000,
                return_messages=True
            )
            
            # Load existing chat history
            chat_history = self.db_manager.get_chat_history(session_id, limit=10)
            for chat in chat_history:
                self.conversation_memories[session_id].chat_memory.add_user_message(chat['user_message'])
                self.conversation_memories[session_id].chat_memory.add_ai_message(chat['ai_response'])
        
        return self.conversation_memories[session_id]
        
        
    
    def get_llm_response(self, query: str, language: str, user_context: Dict, session_id: str) -> str:
        """Get response from Groq Llama model with enhanced context"""
        try:
            # Get conversation memory
            memory = self.get_conversation_memory(session_id)
            
            # Search for relevant context
            relevant_contexts = self.vector_memory.search_relevant_context(query, session_id, n_results=3)
            
            # Create enhanced system prompt
            system_prompt = self.create_enhanced_system_prompt(language, user_context, relevant_contexts, memory)
            
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            
            response = completion.choices[0].message.content
            
            # Add to conversation memory
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}. Please check your Groq API key and try again."
    
    def create_enhanced_system_prompt(self, language: str, user_context: Dict, relevant_contexts: List[str], memory) -> str:
        """Create enhanced system prompt with memory and context"""
        
        # Get conversation summary
        try:
            conversation_summary = memory.predict_new_summary(
                memory.chat_memory.messages, ""
            ) if memory.chat_memory.messages else "No previous conversation."
        except:
            conversation_summary = "No previous conversation."
        
        # Format relevant contexts
        context_text = "\n".join([f"- {ctx[:200]}" for ctx in relevant_contexts[:3]]) if relevant_contexts else "No relevant context found."
        
        base_prompt = """You are Krishi Mitra, an AI farming assistant for Indian farmers. 
        You help with crop advice, fertilizer recommendations, pest management, and irrigation guidance.
        
        User Profile:
        - Location: {location}
        - Primary Crops: {crops}
        - Farm Size: {farm_size}
        
        Conversation Summary:
        {conversation_summary}
        
        Relevant Knowledge:
        {context_text}
        
        Guidelines:
        1. Use the conversation summary to maintain context
        2. Reference relevant knowledge when applicable
        3. Provide practical, actionable advice
        4. Consider Indian farming conditions and practices
        5. Be empathetic and supportive
        6. Ask clarifying questions when needed
        7. Prioritize sustainable farming practices
        8. If you don't know something, admit it and suggest consulting local experts
        """.format(
            location=user_context.get('location', 'Not specified'),
            crops=user_context.get('crops', 'Not specified'),
            farm_size=user_context.get('farm_size', 'Not specified'),
            conversation_summary=conversation_summary,
            context_text=context_text
        )
        
        if language == "Hindi":
            base_prompt += "\n9. Respond primarily in Hindi (Devanagari script)"
        elif language == "HindEnglish":
            base_prompt += "\n9. You can mix Hindi and English as appropriate (Hinglish)"
        else:
            base_prompt += "\n9. Respond in English"
            
        return base_prompt
    
    def initialize_user_session(self, session_id: str):
        """Initialize a new user session with database integration"""
        if session_id not in self.user_sessions:
            # Try to load from database first
            user_profile = self.db_manager.get_user_profile(session_id)
            
            if user_profile:
                self.user_sessions[session_id] = user_profile
            else:
                # Create new session
                self.user_sessions[session_id] = {
                    'session_id': session_id,
                    'location': '',
                    'crops': '',
                    'farm_size': '',
                    'created_at': datetime.now().isoformat(),
                    'last_active': datetime.now().isoformat()
                }
    
    def update_user_profile(self, session_id: str, location: str, crops: str, farm_size: str):
        """Update user profile information with database persistence"""
        self.initialize_user_session(session_id)
        
        # Update in memory
        self.user_sessions[session_id].update({
            'location': location,
            'crops': crops,
            'farm_size': farm_size,
            'last_active': datetime.now().isoformat()
        })
        
        # Save to database
        self.db_manager.save_user_profile(session_id, location, crops, farm_size)
        
        return "Profile updated successfully! ✅ Your farming advice will now be personalized."
    
    def chat_response(self, message: str, history: List, session_id: str) -> Tuple[str, List]:
        """Main chat response function with enhanced memory and query classification"""
        self.initialize_user_session(session_id)
        
        # Detect language
        detected_lang = self.detect_language(message)
        
        # Get user context
        user_context = self.user_sessions[session_id]
        
        # Check if user profile is incomplete
        if not user_context.get('location'):
            return ("🌾 Welcome to Krishi Mitra! \n\n"
                "Please set up your profile first by clicking on the 'Profile Setup' tab. "
                "This helps me provide better farming advice specific to your location and crops.\n\n"
                "कृपया पहले अपनी प्रोफाइल सेट करें।"), history
        
        # Classify query type
        query_type, confidence, analysis = self.query_classifier.classify_query(message, user_context)
        
        # Generate specialized response based on query type
        if query_type == QueryType.FERTILIZER_ADVICE:
            ai_response = self.response_generator.generate_fertilizer_advice(message, user_context, detected_lang)
        elif query_type == QueryType.PEST_MANAGEMENT:
            ai_response = self.response_generator.generate_pest_management(message, user_context, detected_lang)
        elif query_type == QueryType.IRRIGATION_GUIDANCE:
            ai_response = self.response_generator.generate_irrigation_guidance(message, user_context, detected_lang)
        elif query_type == QueryType.CROP_SELECTION:
            ai_response = self.response_generator.generate_crop_selection(message, user_context, detected_lang)
        elif query_type == QueryType.URGENT_ISSUE:
            ai_response = self.response_generator.generate_urgent_response(message, user_context, detected_lang)
        elif query_type == QueryType.GREETING:
            ai_response = self.generate_greeting_response(detected_lang, user_context)
        else:
            # General response with vector search
            relevant_contexts = self.vector_memory.search_relevant_context(message, session_id, n_results=3)
            ai_response = self.response_generator.generate_general_response(message, user_context, detected_lang, relevant_contexts)
        
        # Save to database
        self.db_manager.save_chat_message(session_id, message, ai_response, detected_lang)
        
        # Add to vector memory
        self.vector_memory.add_chat_memory(session_id, message, ai_response, detected_lang)
        
        # Update conversation memory
        memory = self.get_conversation_memory(session_id)
        memory.chat_memory.add_user_message(message)
        memory.chat_memory.add_ai_message(ai_response)
        
        # Update gradio history
        history.append([message, ai_response])
        
        return "", history
    
    def load_chat_history(self, session_id: str) -> List[List[str]]:
        """Load chat history from database for display"""
        self.initialize_user_session(session_id)
        chat_history = self.db_manager.get_chat_history(session_id, limit=20)
        
        gradio_history = []
        for chat in chat_history:
            gradio_history.append([chat['user_message'], chat['ai_response']])
        
        return gradio_history
    
    def generate_greeting_response(self, language: str, user_context: Dict) -> str:
        """Generate personalized greeting response"""
        location = user_context.get('location', '')
        crops = user_context.get('crops', '')
        
        if language == "Hindi":
            return f"नमस्ते! मैं कृषि मित्र हूं। {location} से आने वाले {crops} उगाने वाले किसान का स्वागत है। आपकी खेती में कैसे मदद कर सकता हूं?"
        elif language == "HindEnglish":
            return f"Namaste! Main Krishi Mitra hun. {location} ke {crops} farmer ka welcome hai! Aapki farming mein kaise help kar sakta hun?"
        else:
            return f"Hello! I'm Krishi Mitra, your farming assistant. Welcome {location} farmer growing {crops}! How can I help you with your farming today?"
    
    def get_user_stats(self, session_id: str) -> Dict:
        """Get user statistics and insights"""
        self.initialize_user_session(session_id)
        chat_history = self.db_manager.get_chat_history(session_id)
        
        if not chat_history:
            return {"total_chats": 0, "languages_used": [], "most_common_topics": []}
        
        languages = [chat['language'] for chat in chat_history]
        language_counts = {lang: languages.count(lang) for lang in set(languages)}
        
        # Simple topic detection based on keywords
        topics = []
        for chat in chat_history:
            message = chat['user_message'].lower()
            if any(word in message for word in ['fertilizer', 'खाद', 'urea']):
                topics.append('Fertilizer')
            elif any(word in message for word in ['pest', 'कीट', 'insect']):
                topics.append('Pest Control')
            elif any(word in message for word in ['water', 'irrigation', 'पानी', 'सिंचाई']):
                topics.append('Irrigation')
            elif any(word in message for word in ['crop', 'फसल', 'seed']):
                topics.append('Crop Management')
        
        topic_counts = {topic: topics.count(topic) for topic in set(topics)}
        
        return {
            "total_chats": len(chat_history),
            "languages_used": list(language_counts.keys()),
            "most_common_topics": sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }

# Initialize the main application
krishi_app = KrishiMitra()

# Custom CSS for better appearance
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.profile-section {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.chat-container {
    max-height: 600px;
    overflow-y: auto;
}
"""

# Create Gradio interface
def create_interface():
    with gr.Blocks(css=css, title="Krishi Mitra - Your AI Farming Assistant") as app:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🌾 Krishi Mitra - कृषि मित्र</h1>
            <p>Your AI-powered farming assistant | आपका AI कृषि सहायक</p>
        </div>
        """)
        
        # Session state
        session_id = gr.State(value="default_session")
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Ask me anything about farming! | खेती के बारे में कुछ भी पूछें!")
                
                chatbot = gr.Chatbot(
                    height=400,
                    label="Krishi Mitra Assistant",
                    show_label=True,
                    container=True,
                    scale=1
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your farming question here... | यहाँ अपना खेती का सवाल लिखें...",
                        label="Your Message",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.Examples(
                    examples=[
                        "What fertilizer should I use for wheat crop?",
                        "गेहूं की फसल के लिए कौन सा उर्वरक अच्छा है?",
                        "Meri cotton crop mein pest problem hai, kya karu?",
                        "How much water does rice need per day?",
                        "मिट्टी की जांच कैसे करें?"
                    ],
                    inputs=msg_input,
                    label="Example Questions"
                )
            
            # Profile Setup Tab
            with gr.Tab("👤 Profile Setup"):
                gr.Markdown("### Set up your farming profile | अपनी खेती की जानकारी दें")
                
                with gr.Column(elem_classes="profile-section"):
                    location_input = gr.Textbox(
                        label="Location (District, State) | स्थान (जिला, राज्य)",
                        placeholder="e.g., Pune, Maharashtra"
                    )
                    
                    crops_input = gr.Textbox(
                        label="Primary Crops | मुख्य फसलें",
                        placeholder="e.g., Wheat, Rice, Cotton | गेहूं, धान, कपास"
                    )
                    
                    farm_size_input = gr.Textbox(
                        label="Farm Size | खेत का आकार",
                        placeholder="e.g., 2 acres, 5 hectares"
                    )
                    
                    profile_btn = gr.Button("Save Profile | प्रोफाइल सेव करें", variant="primary")
                    profile_status = gr.Textbox(label="Status", interactive=False)
            
            # Help Tab
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ### How to use Krishi Mitra | कृषि मित्र का उपयोग कैसे करें
                
                **English:**
                1. First, set up your profile with location and crop details
                2. Ask questions in Hindi, English, or mixed (Hinglish)
                3. Get personalized farming advice based on your profile
                4. Topics covered: fertilizers, pest control, irrigation, crop selection
                5. Your chat history is automatically saved and remembered
                
                **हिंदी:**
                1. पहले अपनी प्रोफाइल में स्थान और फसल की जानकारी दें
                2. हिंदी, अंग्रेजी या मिक्स भाषा में सवाल पूछें  
                3. अपनी प्रोफाइल के आधार पर व्यक्तिगत सलाह पाएं
                4. विषय: उर्वरक, कीट नियंत्रण, सिंचाई, फसल चुनाव
                5. आपकी चैट का इतिहास अपने आप सेव हो जाता है
                
                ### Supported Languages | समर्थित भाषाएं
                - English
                - हिंदी (Hindi)
                - Hinglish (Mixed)
                
                ### New Features in Phase 2 | नई सुविधाएं
                - **Persistent Memory**: Remembers your conversations
                - **Smart Context**: Uses previous chats to give better advice
                - **Knowledge Base**: Built-in farming knowledge database
                - **User Analytics**: Track your farming queries and topics
                """)
            
            # Analytics Tab (New)
            with gr.Tab("📊 Analytics"):
                gr.Markdown("### Your Farming Journey | आपकी खेती की यात्रा")
                
                analytics_btn = gr.Button("Load My Stats | मेरे आंकड़े देखें", variant="primary")
                analytics_output = gr.JSON(label="Your Usage Statistics")
                
                with gr.Row():
                    load_history_btn = gr.Button("Load Chat History | चैट इतिहास लोड करें")
                    clear_history_btn = gr.Button("Clear History | इतिहास साफ करें", variant="secondary")
                
                history_display = gr.Chatbot(
                    label="Your Previous Conversations",
                    height=300,
                    show_label=True
                )
                with gr.Tab("❓ Help"):
                    gr.Markdown("""
                ### How to use Krishi Mitra | कृषि मित्र का उपयोग कैसे करें
                
                **English:**
                1. First, set up your profile with location and crop details
                2. Ask questions in Hindi, English, or mixed (Hinglish)
                3. Get personalized farming advice based on your profile
                4. Topics covered: fertilizers, pest control, irrigation, crop selection
                
                **हिंदी:**
                1. पहले अपनी प्रोफाइल में स्थान और फसल की जानकारी दें
                2. हिंदी, अंग्रेजी या मिक्स भाषा में सवाल पूछें  
                3. अपनी प्रोफाइल के आधार पर व्यक्तिगत सलाह पाएं
                4. विषय: उर्वरक, कीट नियंत्रण, सिंचाई, फसल चुनाव
                
                ### Supported Languages | समर्थित भाषाएं
                - English
                - हिंदी (Hindi)
                - Hinglish (Mixed)
                """)
        
        # Event handlers
        def submit_message(message, history, session):
            return krishi_app.chat_response(message, history, session)
        
        def save_profile(location, crops, farm_size, session):
            return krishi_app.update_user_profile(session, location, crops, farm_size)
        
        def load_user_stats(session):
            return krishi_app.get_user_stats(session)
        
        def load_chat_history_display(session):
            return krishi_app.load_chat_history(session)
        
        def clear_chat_history(session):
            # Clear from memory (database will persist)
            if session in krishi_app.conversation_memories:
                krishi_app.conversation_memories[session].clear()
            return [], "Chat history cleared from current session (database history preserved)"
        
        # Connect events
        send_btn.click(
            submit_message,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot]
        )
        
        msg_input.submit(
            submit_message,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot]
        )
        
        profile_btn.click(
            save_profile,
            inputs=[location_input, crops_input, farm_size_input, session_id],
            outputs=[profile_status]
        )
        
        analytics_btn.click(
            load_user_stats,
            inputs=[session_id],
            outputs=[analytics_output]
        )
        
        load_history_btn.click(
            load_chat_history_display,
            inputs=[session_id],
            outputs=[history_display]
        )
        
        clear_history_btn.click(
            clear_chat_history,
            inputs=[session_id],
            outputs=[history_display, profile_status]
        )
    
    return app

def create_krishi_app():
    """Factory function to create KrishiMitra instance"""
    return KrishiMitra()

# 5. Modify create_interface function to accept an optional app instance:
def create_interface(app_instance=None):
    if app_instance is None:
        app_instance = create_krishi_app()

# # Launch the application
# if __name__ == "__main__":
#     print("🌾 Initializing Krishi Mitra Phase 2...")
#     print("📦 Setting up databases and vector stores...")
    
#     app = create_interface()
    
#     print("✅ Krishi Mitra is ready!")
#     print("🚀 Features available:")
#     print("   - Multilingual chat (Hindi/English/HindEnglish)")
#     print("   - Persistent memory with ChromaDB")
#     print("   - SQLite database for user profiles")
#     print("   - LangChain conversation memory")
#     print("   - Built-in farming knowledge base")
#     print("   - User analytics and chat history")
    
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=7861,
#         share=True,
#         debug=True
#     )
    
# At the end of create_interface function, change the launch to:
# if __name__ == "__main__":
#     app = create_interface()
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=int(os.getenv("PORT", 7861)),  # Use environment port
#         share=True,
#         debug=True
#     )