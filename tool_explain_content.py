from langchain.tools import BaseTool
from my_models import MARITACA_SABIA
from my_keys import MARITACA_API_KEY
from langchain_community.chat_models import ChatMaritalk
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import ast

class ToolExplainContent(BaseTool):
    name: str = "tool_explain_content"
    description: str = """
    Utilize essa ferramenta sempre que for solicitado que voce explique um conteudo para pessoas que nao conhecem o assunto.
    
    # Entrada Requeridas
    - 'tema' (str) : Tema principal informado na pergunta do usuário.
    """
    
    return_direct: bool = True
    
    def _run(self, action): 
        action = ast.literal_eval(action)
        theme_param = action.get("theme", "")

        llm =ChatMaritalk(
            api_key=MARITACA_API_KEY,
            model=MARITACA_SABIA
        )
        
        template_explain_content = PromptTemplate(
         template =   """
        Assuma o papel de um professor preocupado com aspectos de didática do usuário.
        
        1. Elabore uma explicação sobre o tema {theme} que seja compreensível por estudantes na fase de conclusão do ensino médio.
        2. Utilize exemplos do cotidiano para tornar a explicação mais fácil. 
        3. Caso sugira algum recurso para apoiar a explicação, lembre-se do cenário e contexto brasileiro.
        4. Caso você apresente um código, seja didático e utilize Python.
        
        # Tema da explicação: {theme}
        """,
        input_variables=["theme"]
        )
        
        chain = template_explain_content | llm | StrOutputParser()
        response = chain.invoke({"theme": theme_param})
        return response
        