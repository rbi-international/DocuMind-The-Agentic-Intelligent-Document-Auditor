from documind.pipeline.prediction import PredictionPipeline

# 1. Initialize the Tool
classifier = PredictionPipeline()

# 2. Sample Data (Real legal text examples)
sample_text_1 = """
This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of laws principles.
"""

sample_text_2 = """
The Employee agrees that during the term of this Agreement and for a period of one (1) year thereafter, they shall not compete with the Company.
"""

# 3. Run Predictions
print("\n" + "="*50)
print(f"Text 1 Prediction: {classifier.predict(sample_text_1)}")
print("-" * 50)
print(f"Text 2 Prediction: {classifier.predict(sample_text_2)}")
print("="*50 + "\n")