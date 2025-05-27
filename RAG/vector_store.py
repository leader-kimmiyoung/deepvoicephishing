import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv 
import argparse

parser = argparse.ArgumentParser(description="스크립트 설명")
parser.add_argument('--size', type=int, default=256, help='chunk의 크기') 
args = parser.parse_args()

# 환경 변수 로드
load_dotenv("requirements(test).env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

def load_chunks_from_csv(size):
    file_path = f"./processed_data/chunks({size}).csv" 
    try:
        # CSV 파일 로드
        df = pd.read_csv(file_path)
        print(f"{size} CSV 파일에서 로드된 청크 수: {len(df)}")
        
        # DataFrame을 Document 객체로 변환
        chunks = []
        for _, row in df.iterrows():
            chunks.append(Document(
                page_content=row['text'],
                metadata={'id': row['id'], 'is_scam': row['is_scam']}
            ))
        
        return chunks
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        return None

def initialize_pinecone():
    """Pinecone을 초기화합니다."""
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    return pc

def create_vector_store(chunks, index_name="voicephising"):
    """청크로부터 벡터 스토어를 생성하고 리트리버를 반환합니다."""
    embeddings = OpenAIEmbeddings()
    
    vectorstore = Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    # 리트리버 생성 - 유사도 임계값 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={ 
            "k": 3,
            "score_threshold": 0.7,  # 코사인 유사도 0.7 이상만 반환
        }
    )   
    
    return vectorstore, retriever

def test_vector_store(size):
    # CSV 파일에서 청크 로드
    chunks = load_chunks_from_csv(size)
    if chunks is None:
        print("청크를 로드할 수 없습니다. 프로그램을 종료합니다.") 
    
    # Pinecone 초기화
    pc = initialize_pinecone()
    
    # 벡터 스토어 생성
    vectorstore, retriever = create_vector_store(chunks)
    
    # 리트리버를 사용한 검색
    test_query = "계좌 이체를 요구하는 통화"
    results = retriever.get_relevant_documents(test_query)
    
    print("\n검색 테스트 결과:")
    for doc in results:
        print("-" * 50)
        print(doc.page_content)
        print(f"메타데이터: {doc.metadata}")
        
test_vector_store(args.size)
