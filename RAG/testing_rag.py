import os 
import re
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv 
from sklearn.metrics import accuracy_score
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse

parser = argparse.ArgumentParser(description="스크립트 설명")
parser.add_argument('--size', type=int, default=256, help='chunk의 크기') 
args = parser.parse_args()

# .env 파일에서 API 키 로드
load_dotenv("requirements.env")

# 프롬프트 템플릿 정의
PROMPT_TEMPLATE = """
다음 통화 내용이 보이스피싱에 해당하는지 논리적으로 판단해주세요.

[보이스피싱 판단 기준]
1. 상대방이 긴박하거나 위급한 상황을 조성하며 행동을 재촉함
2. 본인의 신분을 숨기거나 사칭(경찰, 검사, 금융기관 등)함
3. 금융정보(계좌번호, OTP, 보안카드 등)나 금전 전달을 요구함
4. 대화를 통해 심리적 압박이나 겁을 줌
5. 개인정보를 수집하거나 유도함

[통화 내용 분석 순서]
1. 통화 내용의 핵심 내용을 요약하세요.
2. 보이스피싱 판단 기준 중 어떤 항목과 유사한지 항목별로 비교하세요.
3. 최종적으로 보이스피싱 여부를 판단하고 그 이유를 설명하세요.

[참고할 보이스피싱 사례]
{context}

[분석할 통화 내용]
{question}

[응답 형식]
다음 JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요:
{{
    "segments": [
        {{
            "is_scam": "보이스피싱" 또는 "정상",
            "reason": "판단 이유"
        }}
    ]
}}
"""

# 체인 설정
def set_chain():
    # Pinecone 초기화
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    
    # OpenAI 임베딩 모델 설정
    embeddings = OpenAIEmbeddings()
    
    vectorstore = Pinecone.from_existing_index(
        index_name="voicephising",
        # index_name="scam-pinecone",
        embedding=embeddings
    )
    
    print(f"사용 중인 임베딩 모델: {embeddings.model}")
    
    # 리트리버
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 2  
        }
    )
    
    # ExaOne 3.5 모델 
    llm = Ollama(
        model="exaone3.5:2.4b",
        temperature=0,
        # base_url="http://localhost:11434",
        num_ctx=1024,  
        num_thread=8   
    )
    
    # 프롬프트 
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # RAG 체인
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )
    
    return qa_chain

# 테스트 데이터 로드
def load_test_data(chunk_size):
    scam_csv = pd.read_csv("./dataset(for test)/scam.csv")
    scam_transcript = scam_csv['text'].tolist() 

    normal_csv = pd.read_csv("./dataset(for test)/normal.csv")
    normal_train = normal_csv['text'].tolist()

    transcript = scam_transcript + normal_train

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=10,
    length_function=len,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    chunks = text_splitter.create_documents(transcript)
    
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            'id': i,
            'is_scam': 1,  # 보이스피싱 데이터이므로 1로 설정
            'text': chunk.page_content
        })

    # DataFrame 생성
    chunk_df = pd.DataFrame(chunk_data)
    return chunk_df

def print_result(chunk_df, num):
    text = chunk_df['text'][num]  
    true_label = chunk_df['is_scam'][num]

    result = qa_chain({"query": text})
    content = result["result"]

    try:
        # 마크다운 코드 블록 제거
        content = re.sub(r'```json\s*|\s*```', '', content).strip()
        
        # 정규식으로 직접 필요한 정보 추출 (JSON 파싱 우회)
        is_scam_match = re.search(r'"is_scam":\s*"([^"]*)"', content)
        reason_match = re.search(r'"reason":\s*"([^"]*)"', content, re.DOTALL)
        
        if is_scam_match:
            is_scam_text = is_scam_match.group(1)
            pred = 1 if is_scam_text == "보이스피싱" else 0
        else:
            pred = 1 if "보이스피싱" in content and "정상" not in content else 0
        
        if reason_match:
            reason = reason_match.group(1)
            # 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
            reason = reason.replace('\\n', '\n').replace('\n            ', '\n').strip()
        else:
            reason = "이유 추출 실패"
            
    except Exception as error:
        print(f"파싱 오류: {error}")
        print(f"응답 내용 (첫 200자): {content[:200]}...")
        pred = 1 if "보이스피싱" in content and "정상" not in content else 0
        reason = "파싱 실패"

    print(f"케이스 {num}: 실제 라벨: {'보이스피싱' if true_label == 1 else '정상'}, "
          f"LLM 예측: {'보이스피싱' if pred == 1 else '정상'}")
    print(f"판단 이유: {reason}\n")
    
    return true_label, pred

size = args.size  
qa_chain = set_chain()  # 체인
chunk_df = load_test_data(size)  

true_labels = []  # 실제 라벨
pred_labels = []  # 예측 라벨

for i in range(len(chunk_df)):
    true_label, pred = print_result(chunk_df, i) 
    true_labels.append(true_label)
    pred_labels.append(pred)

log_file = open("result.txt", "a") 

accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n전체 정확도: {accuracy:.3f}")
log_text = f"Input 청크 크기: {size}, 정확도: {accuracy:.3f}"
log_file.write(log_text + "\n")
    
log_file.close()
