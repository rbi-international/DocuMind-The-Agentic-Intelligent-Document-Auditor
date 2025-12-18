# FIX: Import from langchain_core
from langchain_core.tools import Tool
from documind.pipeline.prediction import PredictionPipeline

# Initialize our classification tool
classifier = PredictionPipeline()

def classify_document_tool(text: str) -> str:
    """
    Use this tool to identify the TYPE of a legal document or clause.
    Input: The text of the document.
    Output: The class label (e.g., 'Governing Law', 'Termination').
    """
    return classifier.predict(text)

# Wrap it as a LangChain Tool
tools = [
    Tool(
        name="Document Classifier",
        func=classify_document_tool,
        description="Useful for when you need to know what kind of legal document or clause you are reading. Input should be the text of the clause."
    )
]