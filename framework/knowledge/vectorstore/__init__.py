"""
Author: Nigel J. Navarro
Vector Store
This is the VectorStore interface and its backend implementations
Chroma is the first concrete implementation, but we can add others later.
"""
from framework.knowledge.vectorstore.base import VectorStore
from framework.knowledge.vectorstore.chroma import ChromaVectorStore

# Only imports from the following list will be exposed when using `from vectorstore import *`. 
# This is functionally a "front door" for the vectorstore module, allowing users to access the main classes 
# without needing to know the internal structure of the package.
__all__ = ["VectorStore", "ChromaVectorStore"]