import os

from dotenv import load_dotenv

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from utils.config_loader import load_config

log = CustomLogger().get_logger(__name__)


class ModelLoader:
    """
    Loads the embedding model and the chat LLM described in config/config.yaml.

    The LLM provider is chosen with the LLM_PROVIDER env var (groq | google), so
    switching models is a config change, not a code change. Only the API keys the
    selected providers actually need are required.
    """

    def __init__(self, config: dict | None = None):
        load_dotenv()
        self.config = config or load_config()
        self.provider_key = os.getenv("LLM_PROVIDER", "groq")
        self._validate_env()
        log.info("Configuration loaded", config_keys=list(self.config.keys()), llm_provider=self.provider_key)

    def _required_keys(self) -> list[str]:
        required = {"GOOGLE_API_KEY"}  # embeddings are always Google
        llm_block = self.config.get("llm", {})
        if self.provider_key in llm_block and llm_block[self.provider_key].get("provider") == "groq":
            required.add("GROQ_API_KEY")
        return sorted(required)

    def _validate_env(self):
        required_vars = self._required_keys()
        self.api_keys = {key: os.getenv(key) for key in required_vars}
        missing = [k for k, v in self.api_keys.items() if not v]
        if missing:
            log.error("Missing environment variables", missing_vars=missing)
            raise DocumentPortalException(f"Missing environment variables: {', '.join(missing)}")
        log.info("Environment variables validated", available_keys=required_vars)

    def load_embeddings(self):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=self.api_keys["GOOGLE_API_KEY"])
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", e) from e

    def load_llm(self):
        llm_block = self.config["llm"]
        if self.provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=self.provider_key)
            raise DocumentPortalException(
                f"Provider '{self.provider_key}' not found in config. Options: {', '.join(llm_block)}"
            )

        llm_config = llm_block[self.provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)
        log.info("Loading LLM", provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_keys["GOOGLE_API_KEY"],
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model_name,
                api_key=self.api_keys["GROQ_API_KEY"],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        log.error("Unsupported LLM provider", provider=provider)
        raise DocumentPortalException(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()
    embeddings = loader.load_embeddings()
    print(f"Embedding length: {len(embeddings.embed_query('Hello, how are you?'))}")
    llm = loader.load_llm()
    print(f"LLM Result: {llm.invoke('Hello, how are you?').content}")
