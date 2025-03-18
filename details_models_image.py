from pydantic import BaseModel, Field

from typing import List

class DetailsImageModel(BaseModel):
    title: str = Field(
        description="defina o titulo adequado para a imagem que foi analisada",
    )
    description: str = Field(
        description="Coloque aqui uma descrição detalhada de sua análise para imagem",
    )
    
    rotulos: List[str] = Field(
        description="Defina tres rótulos principais para a imagem analisada",
    )