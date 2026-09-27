from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from model.models import SummaryResponse
from prompt.prompt_library import document_comparison_prompt
from utils.config_loader import load_config


class DocumentComparatorLLM:
    """
    Compare two documents page by page with an LLM and return a validated list of
    {"Page": ..., "Changes": ...} rows (self-healing JSON parsing, like the analyzer).
    """

    def __init__(self, llm=None, max_chars: int | None = None):
        self.log = CustomLogger().get_logger(__name__)
        if llm is None:
            from utils.model_loader import ModelLoader

            llm = ModelLoader().load_llm()
        self.llm = llm
        self.max_chars = max_chars or 2 * load_config().get("analysis", {}).get("max_chars", 60000)
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.chain = document_comparison_prompt | self.llm | self.fixing_parser

    @staticmethod
    def _normalise(response) -> list[dict]:
        """Accept a bare list or a dict wrapping the list under any key; validate every row."""
        if isinstance(response, dict):
            response = next((v for v in response.values() if isinstance(v, list)), [response])
        return [row.model_dump() for row in SummaryResponse.model_validate(response).root]

    def compare_documents(self, combined_docs: str) -> list[dict]:
        try:
            if len(combined_docs) > self.max_chars:
                self.log.info("Truncating comparison input", original_chars=len(combined_docs),
                              max_chars=self.max_chars)
                combined_docs = combined_docs[: self.max_chars]
            response = self.chain.invoke({
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions(),
            })
            rows = self._normalise(response)
            self.log.info("Document comparison completed", rows=len(rows))
            return rows
        except Exception as e:
            self.log.error("Document comparison failed", error=str(e))
            raise DocumentPortalException("Error comparing documents", e) from e


if __name__ == "__main__":
    # Usage: python -m src.doc_compare.retrieval reference.pdf actual.pdf
    import sys
    from pathlib import Path

    from src.doc_compare.data_ingestion import DocumentIngestion

    if len(sys.argv) != 3:
        sys.exit("Usage: python -m src.doc_compare.retrieval <reference.pdf> <actual.pdf>")
    ingestion = DocumentIngestion()
    ref, act = ingestion.save_uploaded_files(Path(sys.argv[1]), Path(sys.argv[2]))
    for row in DocumentComparatorLLM().compare_documents(ingestion.combine_documents(ref, act)):
        print(f"Page {row['Page']}: {row['Changes']}")
