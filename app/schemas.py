from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from .models import JobStatus

class QuestionCreate(BaseModel):
    question_text: str
    reference_answer: str
    max_score: int = 100

class GradingJobCreate(BaseModel):
    job_name: str
    questions: List[QuestionCreate]

class GradingJobResponse(BaseModel):
    id: int
    job_name: str
    status: JobStatus
    total_answers: int
    processed_answers: int
    created_at: datetime
    completed_at: Optional[datetime]
    
class AnswerGradeResponse(BaseModel):
    id: int
    student_id: Optional[str]
    student_answer: str
    model_score: Optional[float]
    human_override_score: Optional[float]
    final_score: Optional[float]
    is_nigerian_variation: Optional[bool]
    
class OverrideRequest(BaseModel):
    answer_id: int
    override_score: float

class EvaluationResponse(BaseModel):
    qwk_all: float
    qwk_standard: float
    qwk_variation: float
    rmse_all: float
    rmse_standard: float
    rmse_variation: float
    total_samples: int
    standard_samples: int
    variation_samples: int
