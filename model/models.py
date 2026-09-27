from pydantic import BaseModel, Field, RootModel


class Metadata(BaseModel):
    """Structured profile returned by the Document Analyzer."""

    Summary: list[str] = Field(default_factory=list, description="Summary of the document as 3-6 bullet points")
    Title: str
    Author: str
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: int | str  # Can be "Not Available"
    SentimentTone: str


class ChangeFormat(BaseModel):
    """One page-level difference between two documents."""

    Page: str = Field(description="Page number (or range) where the change occurs")
    Changes: str = Field(description="Concise description of what changed, or 'NO CHANGE'")


class SummaryResponse(RootModel[list[ChangeFormat]]):
    """List of page-level changes returned by the Document Comparator."""


class ChatAnswer(BaseModel):
    """Answer returned by the RAG chat modules."""

    answer: str
    sources: list[dict] = Field(default_factory=list, description="Retrieved chunks: source, page, snippet")
