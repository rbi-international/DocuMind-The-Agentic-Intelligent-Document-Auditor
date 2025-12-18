import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from documind import logger
import os

class LLMEngine:
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMEngine, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        try:
            logger.info("Initializing Qwen2.5-3B Agent (4-bit Quantization)...")
            
            # SWITCHING TO QWEN (More stable, Native Support)
            model_id = "Qwen/Qwen2.5-3B-Instruct"
            
            # 1. Quantization Config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # 2. Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # 3. Load Model
            # Note: We removed trust_remote_code=True because Qwen is native
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="cuda",
                attn_implementation="eager" # Keeps it stable on Windows
            )

            # 4. Create Pipeline
            text_generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                model_kwargs={"use_cache": False}, # Disable cache explicitly
                temperature=0.1,
                do_sample=True,
                return_full_text=False
            )

            # 5. Wrap in LangChain
            self._pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info("Qwen2.5 Agent is ready.")

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise e

    def get_llm(self):
        # Wrap in Chat Interface
        return ChatHuggingFace(llm=self._pipeline)