content = """pandas
numpy
matplotlib
seaborn
scikit-learn
pyyaml
python-dotenv
ensure
python-box
joblib
types-PyYAML
mlflow
dvc
dvc-s3
dagshub
transformers
accelerate
peft
bitsandbytes
datasets
safetensors
sentencepiece
langchain
langchain-community
langgraph
chromadb
sentence-transformers
fastapi
uvicorn
python-multipart
pydantic
typing-extensions
"""

with open("requirements.txt", "w") as f:
    f.write(content)

print("requirements.txt has been forcefully overwritten with the correct packages.")