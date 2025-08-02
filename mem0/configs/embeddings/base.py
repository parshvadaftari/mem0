from abc import ABC
from typing import Dict, Optional, Union, Callable

import httpx

from mem0.configs.base import AzureConfig


class BaseEmbedderConfig(ABC):
    """
    Config for Embeddings.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[Union[str, Callable[[], str]]] = None,
        embedding_dims: Optional[int] = None,
        # Ollama specific
        ollama_base_url: Optional[str] = None,
        # Openai specific
        openai_base_url: Optional[str] = None,
        # Huggingface specific
        model_kwargs: Optional[dict] = None,
        huggingface_base_url: Optional[str] = None,
        # AzureOpenAI specific
        azure_kwargs: Optional[AzureConfig] = {},
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # VertexAI specific
        vertex_credentials_json: Optional[str] = None,
        memory_add_embedding_type: Optional[str] = None,
        memory_update_embedding_type: Optional[str] = None,
        memory_search_embedding_type: Optional[str] = None,
        # Gemini specific
        output_dimensionality: Optional[str] = None,
        # LM Studio specific
        lmstudio_base_url: Optional[str] = "http://localhost:1234/v1",
        # AWS Bedrock specific
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = "us-west-2",
    ):
        """
        Initializes a configuration class instance for the Embeddings.

        :param model: Embedding model to use, defaults to None
        :type model: Optional[str], optional
        :param api_key: API key to be use, can be a string or a callable that returns a string, defaults to None
        :type api_key: Optional[Union[str, Callable[[], str]]], optional
        :param embedding_dims: The number of dimensions in the embedding, defaults to None
        :type embedding_dims: Optional[int], optional
        :param ollama_base_url: Base URL for the Ollama API, defaults to None
        :type ollama_base_url: Optional[str], optional
        :param model_kwargs: key-value arguments for the huggingface embedding model, defaults a dict inside init
        :type model_kwargs: Optional[Dict[str, Any]], defaults a dict inside init
        :param huggingface_base_url: Huggingface base URL to be use, defaults to None
        :type huggingface_base_url: Optional[str], optional
        :param openai_base_url: Openai base URL to be use, defaults to "https://api.openai.com/v1"
        :type openai_base_url: Optional[str], optional
        :param azure_kwargs: key-value arguments for the AzureOpenAI embedding model, defaults a dict inside init
        :type azure_kwargs: Optional[Dict[str, Any]], defaults a dict inside init
        :param http_client_proxies: The proxy server settings used to create self.http_client, defaults to None
        :type http_client_proxies: Optional[Dict | str], optional
        :param vertex_credentials_json: The path to the Vertex AI credentials JSON file, defaults to None
        :type vertex_credentials_json: Optional[str], optional
        :param memory_add_embedding_type: The type of embedding to use for the add memory action, defaults to None
        :type memory_add_embedding_type: Optional[str], optional
        :param memory_update_embedding_type: The type of embedding to use for the update memory action, defaults to None
        :type memory_update_embedding_type: Optional[str], optional
        :param memory_search_embedding_type: The type of embedding to use for the search memory action, defaults to None
        :type memory_search_embedding_type: Optional[str], optional
        :param lmstudio_base_url: LM Studio base URL to be use, defaults to "http://localhost:1234/v1"
        :type lmstudio_base_url: Optional[str], optional
        :param aws_access_key_id: AWS access key ID, defaults to None
        :type aws_access_key_id: Optional[str], optional
        :param aws_secret_access_key: AWS secret access key, defaults to None
        :type aws_secret_access_key: Optional[str], optional
        :param aws_region: AWS region, defaults to "us-west-2"
        :type aws_region: Optional[str], optional
        """

        self.model = model
        self.api_key = api_key
        self.openai_base_url = openai_base_url
        self.embedding_dims = embedding_dims

        # AzureOpenAI specific
        self.http_client = httpx.Client(proxies=http_client_proxies) if http_client_proxies else None

        # Ollama specific
        self.ollama_base_url = ollama_base_url

        # Huggingface specific
        self.model_kwargs = model_kwargs or {}
        self.huggingface_base_url = huggingface_base_url
        # AzureOpenAI specific
        self.azure_kwargs = AzureConfig(**azure_kwargs) or {}

        # VertexAI specific
        self.vertex_credentials_json = vertex_credentials_json
        self.memory_add_embedding_type = memory_add_embedding_type
        self.memory_update_embedding_type = memory_update_embedding_type
        self.memory_search_embedding_type = memory_search_embedding_type

        # Gemini specific
        self.output_dimensionality = output_dimensionality

        # LM Studio specific
        self.lmstudio_base_url = lmstudio_base_url

        # AWS Bedrock specific
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region

    def get_api_key(self) -> Optional[str]:
        """
        Resolve the API key, handling both string and callable values.
        
        Returns:
            Optional[str]: The resolved API key string, or None if not available
        """
        if self.api_key is None:
            return None
        
        if callable(self.api_key):
            return self.api_key()
        else:
            return self.api_key
