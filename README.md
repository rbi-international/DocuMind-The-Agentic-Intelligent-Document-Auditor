# DocuMind: Enterprise Agentic AI for Legal Document Auditing

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?style=for-the-badge&logo=pytorch)
![LangGraph](https://img.shields.io/badge/Agentic-LangGraph-FF4B4B?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-grey?style=for-the-badge)

**DocuMind** is an end-to-end **Agentic AI System** capable of autonomously reading, classifying, and auditing legal contracts. Unlike simple wrappers around ChatGPT, this project implements a **Hybrid AI Architecture** running locally on consumer hardware (NVIDIA RTX 3060).

It combines a **Fine-Tuned Discriminative Model** (DistilBERT) for high-precision classification with a **Quantized Generative Model** (Qwen2.5-3B) for reasoning and risk analysis, orchestrated via **LangGraph**.

---

## 🧠 The "Why": Agentic AI vs. The Rest

Why build an Agent instead of just using a standard Classifier, RAG, or an API?

### 1. Why not just a Standard Classifier (CNN/BERT)?
*   **Limitation:** A standard model (like ResNet or BERT) can only output a label (e.g., "Non-Disclosure Agreement"). It cannot read the text to find *loopholes*, *risks*, or *anomalies*.
*   **DocuMind Advantage:** Our system identifies the document type **AND** reads it to explain *why* it is risky.

### 2. Why not RAG (Retrieval Augmented Generation)?
*   **Limitation:** RAG is great for finding facts ("What is the termination date?"). However, RAG relies on the LLM's general knowledge. It doesn't inherently understand niche legal structures without specific training.
*   **DocuMind Advantage:** We utilize a **Fine-Tuned Tool**. We trained a specific model on the **LEDGAR** dataset (Legal Provisions) to understand legal jargon mathematically. The Agent uses this specialized tool, resulting in higher domain accuracy than a generic RAG system.

### 3. Why not just call OpenAI/Claude API?
*   **Limitation:**
    1.  **Data Privacy:** Sending sensitive legal contracts to a public cloud API is a security violation in most MNCs.
    2.  **Cost:** Processing thousands of documents via API is expensive.
    3.  **Latency:** Cloud APIs introduce network lag.
*   **DocuMind Advantage:** **100% Local Execution.** By using 4-bit Quantization (`bitsandbytes`), we run a massive 3-Billion parameter brain on a 6GB Laptop GPU. Zero data leaves the machine.

---

## 🏗 System Architecture

The project follows the **ReAct (Reason + Act)** pattern enabled by **LangGraph**.

### The "Brain" (Orchestrator)
*   **Model:** Qwen2.5-3B-Instruct (Quantized to NF4).
*   **Role:** Acts as the Project Manager. It receives the user request, decides which tools to use, interprets the tool's output, and formulates the final answer.

### The "Hands" (Custom Tools)
*   **Model:** DistilBERT-base-uncased.
*   **Training:** Fine-tuned for 3 epochs on the **LEDGAR** dataset (~60,000 legal provisions).
*   **Role:** A highly specialized tool that classifies text into 100+ legal categories (e.g., *Governing Law*, *Indemnification*, *Waivers*) with high precision.

### The Pipeline
1.  **Ingestion:** User uploads text via **Streamlit**.
2.  **Dispatch:** **FastAPI** sends text to the **LangGraph Agent**.
3.  **Reasoning Loop:**
    *   Agent analyzes text.
    *   Agent calls `Document Classifier` Tool.
    *   Tool performs inference (CPU/GPU) and returns label.
    *   Agent interprets label + original text to assess risk.
4.  **Response:** Structured JSON output returned to UI.

---

## 🛠 Tech Stack & MLOps

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Core logic. |
| **DL Framework** | PyTorch (CUDA 12.1) | Tensor operations & GPU acceleration. |
| **LLM Engine** | HuggingFace Transformers | Loading Qwen2.5 & DistilBERT. |
| **Optimization** | BitsAndBytes (NF4) | 4-bit Quantization to fit 3B model in 6GB VRAM. |
| **Agent Framework** | LangGraph | State-machine based Agent orchestration. |
| **API Backend** | FastAPI | Asynchronous microservice architecture. |
| **Frontend** | Streamlit | Rapid UI prototyping. |
| **Experiment Tracking** | MLflow | Logging metrics (Accuracy, Loss) during training. |
| **Data Versioning** | DVC | Tracking changes in datasets. |

---

## 📂 Project Structure

The project follows a modular, package-based structure suitable for PyPI distribution.

```text
DocuMind-Agent/
├── artifacts/                  # [Auto-Generated] Stores trained models & data
├── config/                     # Configuration files
│   └── config.yaml             # Single source of truth for paths/params
├── src/
│   └── documind/
│       ├── components/         # Business Logic (ModelTrainer, LLMEngine)
│       ├── pipeline/           # Orchestrators (Training, Prediction, Agent)
│       ├── entity/             # Data Classes (Pydantic/Dataclasses)
│       └── utils/              # Common helpers
├── app.py                      # FastAPI Backend Entry Point
├── streamlit_app.py            # Streamlit Frontend Entry Point
├── main.py                     # Training Pipeline Entry Point
├── dvc.yaml                    # DVC Pipeline definitions
├── requirements.txt            # Project Dependencies
└── pyproject.toml              # Build System Requirements
🚀 Installation & Setup
Prerequisites
OS: Windows (with WSL2 recommended) or Linux.
GPU: NVIDIA GPU with at least 6GB VRAM.
Drivers: CUDA 12.1 or compatible.
1. Clone & Environment

```bash
git clone https://github.com/rbi-international/DocuMind-The-Agentic-Intelligent-Document-Auditor.git
cd DocuMind-The-Agentic-Intelligent-Document-Auditor
```

# Create a clean Python 3.11 environment
```bash
conda create -n documind python=3.11 -y
conda activate documind
```
2. Install Dependencies
We separate PyTorch installation to ensure the correct CUDA version is used.

```Bash
# 1. Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install Project Requirements
pip install -r requirements.txt

# 3. Install Local Package
pip install -e .
```
🧠 Training the Specialist Tool
Before the Agent can work, the DistilBERT tool must be trained.

```Bash
python main.py
This executes the MLOps pipeline:
Ingestion: Downloads dataset from HuggingFace.
Validation: Validates schema/columns.
Transformation: Tokenizes text.
Training: Fine-tunes DistilBERT (Approx 5-10 mins on RTX 3060).
Evaluation: Logs accuracy to MLflow.
🏃‍♂️ Running the Application
The system requires two terminals to simulate a Microservices architecture.
Terminal 1: The Backend (FastAPI)
This loads the AI models into GPU memory.
code
Bash
conda activate documind
python app.py
Status: API will run on http://0.0.0.0:8000
Terminal 2: The Frontend (Streamlit)
This launches the user interface.
code
Bash
conda activate documind
streamlit run streamlit_app.py
Status: Dashboard will open at http://localhost:8501
🔮 Future Enhancements (Roadmap)
To take this from "Project" to "Product", the following steps are planned:
Vector Database (RAG): Integrate ChromaDB to allow the Agent to "Look up" past contracts for comparison.
Dockerization: Create a Dockerfile and docker-compose.yml to orchestrate the Backend and Frontend containers.
CI/CD: Implement GitHub Actions to auto-run tests (pytest) upon push.
Model Drift Monitoring: Use Evidently AI to detect if incoming legal documents differ significantly from the training data (LEDGAR).
Quantization Optimization: Explore AWQ or GPTQ for faster inference speeds on edge devices.
📜 License
This project is licensed under the MIT License.
Author: [Rohit Bharti]
Contact: [rohit.bharti8211@gmail.com]
LinkedIn: [https://www.linkedin.com/in/rohitbharti13/]