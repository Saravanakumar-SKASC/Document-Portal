import sys
from pathlib import Path

from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from model.models import Metadata
from prompt.prompt_library import document_analysis_prompt
from utils.config_loader import load_config


class DocumentAnalyzer:
    """
    Extracts structured metadata and a summary from document text.

    The LLM's JSON is validated against the ``Metadata`` schema; if the model
    returns malformed JSON, an OutputFixingParser asks the LLM to repair it.
    Pass ``llm`` to inject a model (used by the tests); otherwise it comes from config.
    """

    def __init__(self, llm=None, max_chars: int | None = None):
        self.log = CustomLogger().get_logger(__name__)
        try:
            if llm is None:
                from utils.model_loader import ModelLoader

                llm = ModelLoader().load_llm()
            self.llm = llm
            self.max_chars = max_chars or load_config().get("analysis", {}).get("max_chars", 60000)
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            self.prompt = document_analysis_prompt
            self.log.info("DocumentAnalyzer initialized successfully")
        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("Error initializing DocumentAnalyzer", error=str(e))
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", e) from e

    def analyze_document(self, document_text: str) -> dict:
        """Analyze a document's text and extract structured metadata & summary."""
        try:
            if not document_text or not document_text.strip():
                raise DocumentPortalException("Document is empty; nothing to analyze.")
            if len(document_text) > self.max_chars:
                self.log.info("Truncating document for analysis", original_chars=len(document_text),
                              max_chars=self.max_chars)
                document_text = document_text[: self.max_chars]

            chain = self.prompt | self.llm | self.fixing_parser
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text,
            })
            self.log.info("Metadata extraction successful", keys=list(response.keys()))
            return response
        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed", e) from e


if __name__ == "__main__":
    # Usage: python -m src.document_analyzer.data_analysis path/to/file.pdf
    from src.document_analyzer.data_ingestion import DocumentHandler

    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.document_analyzer.data_analysis <path-to-pdf>")
    handler = DocumentHandler()
    text = handler.read_pdf(handler.save_pdf(Path(sys.argv[1])))
    for key, value in DocumentAnalyzer().analyze_document(text).items():
        print(f"{key}: {value}")
