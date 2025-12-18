from documind.pipeline.agent_pipeline import AgentPipeline
from documind import logger

# Sample text: A Governing Law clause
text = """
This Agreement shall be governed by and construed in accordance with the laws of the State of California, 
without giving effect to any choice of law or conflict of law provisions.
"""

try:
    logger.info(">>> STARTING AGENT TEST <<<")
    
    # Initialize our custom pipeline
    agent = AgentPipeline()
    
    # Run the agent
    result = agent.run_agent(text)
    
    print("\n" + "="*50)
    print("FINAL AGENT OUTPUT:")
    print(result)
    print("="*50 + "\n")

except Exception as e:
    logger.exception(e)