from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Enum, Boolean
from sqlalchemy.sql import func
from .database import Base
import enum

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class GradingJob(Base):
    __tablename__ = "grading_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String(255), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    total_answers = Column(Integer, default=0)
    processed_answers = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("grading_jobs.id"))
    question_text = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=False)
    max_score = Column(Integer, default=100)
    
class StudentAnswer(Base):
    __tablename__ = "student_answers"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("grading_jobs.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    student_id = Column(String(100), nullable=True)
    student_answer = Column(Text, nullable=False)
    model_score = Column(Float, nullable=True)
    human_override_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    grading_time_ms = Column(Integer, nullable=True)
    is_nigerian_variation = Column(Boolean, default=False)

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(255), nullable=False)
    qwk_all = Column(Float, nullable=True)
    qwk_standard = Column(Float, nullable=True)
    qwk_variation = Column(Float, nullable=True)
    rmse_all = Column(Float, nullable=True)
    rmse_standard = Column(Float, nullable=True)
    rmse_variation = Column(Float, nullable=True)
    total_samples = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
