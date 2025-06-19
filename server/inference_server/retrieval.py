from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from pinecone import Pinecone as PineconeClient
import os


# 체인 설정
class Retrieval:
    def __init__(self):
        # Pinecone 초기화
        self.pineconeClient = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])
        # OpenAI 임베딩 모델 설정
        self.embeddings = OpenAIEmbeddings()
        # 벡터 스토어
        self.vectorstore = Pinecone.from_existing_index(
            index_name="voicephising",
            embedding=self.embeddings
        )
        # 리트리버
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 2
            }
        )

    def retrieve(self, query):
        if self.retriever is None:
            raise ValueError("Retriever is not initialized. Call set_chain() first.")
        documents = self.retriever.get_relevant_documents(query)
        return [doc.page_content for doc in documents]









