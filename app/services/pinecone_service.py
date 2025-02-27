from pinecone import Pinecone, ServerlessSpec
from app.config import get_settings

settings = get_settings()

class PineconeService:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.index_name)

    async def query_documents(self, query: str, namespace: str = None):
        try:
            query_response = self.index.query(
                vector=query,
                top_k=3,
                include_metadata=True,
                namespace=namespace
            )
            return query_response
        except Exception as e:
            raise Exception(f"Pinecone query failed: {str(e)}")

    def extract_context(self, query_response):
        contexts = []
        references = []

        for match in query_response.matches:
            if hasattr(match.metadata, 'text'):
                contexts.append(match.metadata.text)
            if hasattr(match.metadata, 'original_filename'):
                references.append(match.metadata.original_filename)

        return "\n".join(contexts), references
