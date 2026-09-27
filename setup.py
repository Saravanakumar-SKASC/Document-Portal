from setuptools import find_packages, setup

setup(
    name="document_portal",
    author="Saravanakumar N",
    version="1.0.0",
    description="LLM document intelligence: analyze, chat (RAG) and compare documents",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests", "tests.*"]),
)
