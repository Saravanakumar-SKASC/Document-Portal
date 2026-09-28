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

    def _provider_blocks(self) -> dict:
        return {k: v for k, v in self.config["llm"].items() if isinstance(v, dict) and "provider" in v}

    def _build_llm(self, provider_key: str):
        blocks = self._provider_blocks()
        if provider_key not in blocks:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise DocumentPortalException(
                f"Provider '{provider_key}' not found in config. Options: {', '.join(blocks)}"
            )

        llm_config = blocks[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)
        max_retries = self.config["llm"].get("max_retries", 2)
        timeout = self.config["llm"].get("timeout_seconds", 60)
        log.info("Loading LLM", provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                max_retries=max_retries,
                timeout=timeout,
            )
        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                request_timeout=timeout,
            )

        log.error("Unsupported LLM provider", provider=provider)
        raise DocumentPortalException(f"Unsupported LLM provider: {provider}")

    def _fallback_provider_key(self) -> str | None:
        """The other configured provider, if its API key is available."""
        key_for = {"google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}
        for name, block in self._provider_blocks().items():
            if name != self.provider_key and os.getenv(key_for.get(block["provider"], "")):
                return name
        return None

    def load_llm(self, with_fallback: bool = True):
        """
        Primary LLM (LLM_PROVIDER). Each call already retries with backoff; if the
        provider still fails (outage, rate limit, timeout) the request is re-run on
        the other provider, so one vendor outage doesn't take the assistant down.
        """
        primary = self._build_llm(self.provider_key)
        fallback_key = self._fallback_provider_key() if with_fallback else None
        if fallback_key:
            log.info("LLM fallback enabled", primary=self.provider_key, fallback=fallback_key)
            return with_fallback_llm(primary, self._build_llm(fallback_key))
        return primary


def with_fallback_llm(primary, fallback):
    """Wrap ``primary`` so any exception re-runs the same input on ``fallback``."""
    return primary.with_fallbacks([fallback])


if __name__ == "__main__":
    loader = ModelLoader()
    embeddings = loader.load_embeddings()
    print(f"Embedding length: {len(embeddings.embed_query('Hello, how are you?'))}")
    llm = loader.load_llm()
    print(f"LLM Result: {llm.invoke('Hello, how are you?').content}")
