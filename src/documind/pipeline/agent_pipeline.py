from documind.components.llm_engine import LLMEngine
from documind.components.agent_tools import tools
from documind import logger
from langchain_core.messages import HumanMessage
# Ensure this is the import line:
from langgraph.prebuilt import create_react_agent 

class AgentPipeline:
    def __init__(self):
        # 1. Get the LLM (Now it returns a ChatModel)
        self.llm = LLMEngine().get_llm()
        
        # 2. Create the Agent using LangGraph
        self.agent = create_react_agent(self.llm, tools)
        
    def run_agent(self, document_text: str):
        try:
            logger.info("Initializing Agentic Workflow (LangGraph)...")
            
            # 3. Construct Input
            user_input = f"""
            Task: Classify this legal text and find the risk.
            Text: "{document_text}"
            First, use the 'Document Classifier' tool.
            Then, summarize the result.
            """
            
            messages = [HumanMessage(content=user_input)]

            # 4. Run Graph
            logger.info("Agent is thinking...")
            result = self.agent.invoke({"messages": messages})
            
            # 5. Extract Answer
            final_response = result["messages"][-1].content
            return final_response

        except Exception as e:
            logger.error(f"Agent failed: {e}")
            return "Agent Error - Check logs."