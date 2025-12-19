from fastapi import FastAPI, HTTPException, BackgroundTasks
from documind.pipeline.agent_pipeline import AgentPipeline
from documind.entity.api_models import DocumentRequest, AuditResponse
from documind import logger
import uvicorn
import os

# 1. Initialize API
app = FastAPI(
    title="DocuMind API",
    description="Agentic AI for Legal Document Auditing (Qwen-2.5 + DistilBERT)",
    version="1.0.0"
)

# 2. Global Variable for the Agent (Singleton Pattern)
# We load it purely to None first. We will load it on "startup".
agent_pipeline = None

@app.on_event("startup")
async def startup_event():
    """
    Load the Heavy AI Models when the server starts.
    This prevents loading them for every single request.
    """
    global agent_pipeline
    try:
        logger.info(">>> STARTUP: Loading AI Models into GPU... <<<")
        agent_pipeline = AgentPipeline()
        logger.info(">>> STARTUP: Models Loaded Successfully! <<<")
    except Exception as e:
        logger.error(f"Failed to load AI Models: {e}")
        raise e

@app.get("/")
async def root():
    return {"status": "Online", "message": "DocuMind API is running. Go to /docs for Swagger UI."}

@app.post("/audit", response_model=AuditResponse)
async def audit_document(request: DocumentRequest):
    """
    Endpoint to audit a legal document text.
    """
    global agent_pipeline
    
    if not agent_pipeline:
        raise HTTPException(status_code=503, detail="AI Model not loaded.")

    if len(request.text) < 10:
        raise HTTPException(status_code=400, detail="Text too short. Please provide a valid legal clause.")

    try:
        logger.info("Received audit request...")
        
        # Run the Agent
        # Note: In production, we might offload this to Celery, but for now we await it.
        result_text = agent_pipeline.run_agent(request.text)
        
        # Return structured response
        # Since our Agent output is text, we map it to 'raw_agent_output'
        # Ideally, we would ask the agent to output JSON, but text is fine for now.
        return AuditResponse(
            risk_analysis="Processed by Agent", 
            raw_agent_output=result_text
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Host 0.0.0.0 allows access from other machines/docker
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)