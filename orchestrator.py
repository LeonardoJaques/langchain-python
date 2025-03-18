from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY 
from langchain.globals import set_debug

set_debug(False)

from langchain import hub
from langchain.agents import create_react_agent
from langchain.agents import Tool
from tool_analyze_image import ToolAnalyzeImage
from tool_explain_content import ToolExplainContent

class OrchestratorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_FLASH)
        tool_analyze_image = ToolAnalyzeImage()
        tool_explain_content = ToolExplainContent()
        
        self.tools = [
            Tool(
                name= tool_analyze_image.name,
                func= tool_analyze_image.run,
                description = tool_analyze_image.description,
                return_direct = tool_analyze_image.return_direct
            ),
            Tool(
                name= tool_explain_content.name,
                func= tool_explain_content.run,
                description = tool_explain_content.description,
                return_direct = tool_explain_content.return_direct
            )
        ]
        
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, self.tools, prompt)
    
    
    
       