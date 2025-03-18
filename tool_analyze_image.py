from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from details_models_image import DetailsImageModel
import ast

class ToolAnalyzeImage(BaseTool):
    name: str = "Analyze Image"
    description: str = """ 
    Utilize esta ferramenta sempre que for solicitado que voce faça uma análise de uma imagem.
    
    # Entradas Requeridas
    - 'image_name' (str) : Nome da imagem a ser analisada com extensão de JPG.
    Example: 'Teste.jpge' ou 'Teste.jpg' 
    """
    return_direct: bool = False
    
    def _run(self, action):
        action = ast.literal_eval(action)
        path_image = action.get("image_name","")
        
        llm = ChatGoogleGenerativeAI(
        api_key= GEMINI_API_KEY,
        model= GEMINI_FLASH 
        )

        # Example of analyzing an image

        image = encode_image(f"data/{path_image}")

        template_analyze = ChatPromptTemplate.from_messages([
        ("system",
        """ 
        Assuma que você é um analisador de imagens. 
        A sua tarefa principal consiste em: analisar imagens e extrair informações importantes de forma objetiva e clara

        # Formato de saída
        Descrição da imagem: 'Coloque sua descrição da imagem aqui'
        Rótulos: 'Coloque uma lista com três termos chave separados por vírgula'
        
        """
        ),
        ("user",
        [
                {
                "type": "text",
                "text": "Descreva a imagem"
                },
                {
                "type": "image_url",
                "image_url": {"url":"data:image/jpg;base64,{image_data}"}
                }
            ]
        )
        ])

        chain_image_analyze = template_analyze | llm | StrOutputParser()

        parser_json_image = JsonOutputParser(
        pydantic_object=DetailsImageModel
        )

        template_response = PromptTemplate(
        template="""
            Gere um resumo, utilizando uma linguagem clara e objetiva, focado no principal brasileiro. 
            A ideia é que a comunicação do resultado seja o mais fácil possível, priorizando registros para consulta posteriores.
            
            # O resultado da análise da imagem é:
            {response_analyze_image_chain}
            
            # output_format
            {output_format}
            
        """,
        input_variables={"response_analyze_image_chain"},
        partial_variables={
            "output_format": parser_json_image.get_format_instructions()
        }
        )


        chain_summary = template_response | llm | parser_json_image

        chain_complex = (chain_image_analyze | chain_summary)

        response = chain_complex.invoke({"image_data": image})   
        return response