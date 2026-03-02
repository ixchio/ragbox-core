"""
Zero-Configuration Settings via Pydantic using dotenv.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    RAGBox Configuration.
    Prioritizes Env Vars, falls back to .env if present, otherwise defaults.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API Key")
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API Key")

    # Local paths
    local_model_path: str = Field(default="models/llama-3.1-8b-instruct.gguf", description="Path for local LLM")
    chroma_db_dir: str = Field(default="./chroma_db", description="Path for local vector DB")
    
    # Behaviors
    log_level: str = "INFO"
    max_agent_steps: int = 5
    graph_community_level: int = 1
