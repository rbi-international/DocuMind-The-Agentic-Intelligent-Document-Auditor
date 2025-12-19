from pydantic import BaseModel

class DocumentRequest(BaseModel):
    text: str

class AuditResponse(BaseModel):
    filename: str = "input_text"
    classification: str = "Unknown"
    risk_analysis: str
    raw_agent_output: str