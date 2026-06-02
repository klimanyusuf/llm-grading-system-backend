import os
import time
import re
from typing import Tuple, Optional

class GradingEngine:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str = "models/llama-3-8b-q4.gguf"):
        """Load the quantized LLM model"""
        self._mock_mode = True
        
        try:
            if os.path.exists(model_path):
                import llama_cpp
                self._model = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_threads=4,
                    n_gpu_layers=35
                )
                self._mock_mode = False
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model not found at {model_path}. Using mock mode for testing.")
        except Exception as e:
            print(f"Failed to load model: {e}. Using mock mode.")
            self._mock_mode = True
    
    def _build_prompt(self, question: str, reference_answer: str, student_answer: str, max_score: int) -> str:
        """Build the grading prompt for the LLM with Nigerian English awareness"""
        return f"""You are an examiner grading short answer questions for Nigerian university students.

IMPORTANT: Nigerian students may write English with natural local variations (e.g., "computer get three parts" instead of "computer has three parts", "CPU dey process" instead of "CPU processes"). Do NOT penalize these variations. Grade based on FACTUAL CORRECTNESS, not grammar style.

Question: {question}
Reference Answer: {reference_answer}
Student Answer: {student_answer}
Maximum Score: {max_score}

Instructions:
- If the student answer contains the key factual information, give full credit
- Ignore grammatical variations that don't change meaning
- Accept common Nigerian English patterns ("get" for "have", "dey" for "is/are", "na" for emphasis)
- Return ONLY an integer score between 0 and {max_score}
- No explanations, no extra text

Score:"""
    
    def _extract_score(self, raw_output: str, max_score: int) -> int:
        """Extract integer score from LLM output"""
        numbers = re.findall(r'\b(\d+)\b', raw_output)
        if numbers:
            score = int(numbers[0])
            return max(0, min(score, max_score))
        return max_score // 2
    
    def grade(self, question: str, reference_answer: str, student_answer: str, max_score: int = 100) -> Tuple[int, float]:
        """Grade a single student answer. Returns: (score, inference_time_ms)"""
        start_time = time.time()
        
        if self._mock_mode:
            # Mock grading for local testing without GPU
            student_lower = student_answer.lower()
            ref_lower = reference_answer.lower()
            
            ref_words = set(ref_lower.split())
            student_words = set(student_lower.split())
            
            overlap = len(ref_words.intersection(student_words))
            base_score = (overlap / max(len(ref_words), 1)) * max_score
            
            nigerian_patterns = ['dey', 'get', 'na', 'wey', 'sef', 'abeg', 'bros', 'sis']
            has_nigerian_pattern = any(pattern in student_lower for pattern in nigerian_patterns)
            
            if has_nigerian_pattern and base_score < max_score * 0.7:
                base_score = max(base_score, max_score * 0.7)
            
            score = int(min(max(base_score, 0), max_score))
            elapsed_ms = int((time.time() - start_time) * 1000)
            return score, elapsed_ms
        
        prompt = self._build_prompt(question, reference_answer, student_answer, max_score)
        
        try:
            response = self._model(
                prompt,
                max_tokens=10,
                temperature=0.1,
                stop=["\n"],
                echo=False
            )
            raw_output = response["choices"][0]["text"].strip()
            score = self._extract_score(raw_output, max_score)
        except Exception as e:
            print(f"Grading error: {e}")
            score = max_score // 2
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        return score, elapsed_ms
    
    def detect_nigerian_variation(self, text: str) -> bool:
        """Detect if text contains Nigerian English variations (for RQ2)"""
        patterns = ['dey', 'get', 'na', 'wey', 'sef', 'abeg', 
                    'bros', 'sis', 'walahi', 'wallahi', 'abi', 'shey']
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in patterns)

grading_engine = GradingEngine()
