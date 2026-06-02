2. Create virtual environment
bash
# Windows
python -m venv venv
venv\Scripts\activate

# LLM Grading System - Backend

FastAPI backend for automatic short answer grading using Llama 3, robust to Nigerian English variations.

## Features

- REST API for grading jobs
- Llama 3 8B integration via Groq API
- Nigerian English variation detection
- CSV upload and export
- Manual grade override
- Evaluation metrics (QWK, RMSE)

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Groq API key (free) - [Get one here](https://console.groq.com)

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/klimanyusuf/llm-grading-system-backend.git
cd llm-grading-system-backend

```
2. Create a Virtual enviroment
# Mac/Linux/windows
python -m venv venv
source venv/bin/activate

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Set up environment variables
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key_here
5. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```
6. Test the API
Open your browser and go to: http://localhost:8000

You should see: {"message":"LLM Grading System","status":"running","mock_mode":false}

API Endpoints

Method	 Endpoint	           Description

POST	/api/jobs	Create a grading job

POST	/api/jobs/{id}/upload	Upload CSV with student answers

POST	/api/jobs/{id}/grade	Start automated grading

GET	/api/jobs/{id}/results	Get grading results

PUT	/api/override	Manually override a grade

GET	/api/jobs/{id}/export	Export results as CSV

GET	/api/evaluate/{id}	Run evaluation (QWK, RMSE)



CSV Format
Your CSV file should have at least these columns:

student_id,answer

CIT001,The CPU is the brain of the computer.

CIT002,CPU dey process all instructions.

Docker Deployment
bash
docker build -t llm-grading-backend .
docker run -p 8000:8000 llm-grading-backend
Live Demo
The backend is deployed at: https://klimanyusuf-llm-grading-system.hf.space

License
MIT
