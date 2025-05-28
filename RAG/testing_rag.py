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
You are a voice phishing detection expert.

[Voice Phishing Clues – If ANY of these appear, classify as phishing]
1. Pretending to be the police, prosecutor’s office, Financial Supervisory Service, National Tax Service, etc.
2. Urgent language like: “right now,” “within today,” “urgent,” “immediate action”
3. Requests for money or personal info: “please send your account number,” “we need your password,” “send money now”
4. Threats or fear language: “involved in a crime,” “your account will be frozen,” “your assets will be seized”

[Normal Call Characteristics]
- Talks about deposits, loans, interest rates, and financial products
- Responds politely to customer inquiries
- No threats or requests for money/personal info

[Reference examples]
{context}

[Call to analyze]
{question}

If the call shows any phishing clue, answer “Phishing”. If not, answer “Normal”.

Answer in JSON format:
{{
  "segments": [
    {{
      "is_scam": "Phishing" or "Normal",
      "reason": "Reason: Brief explanation"
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
        model = "llama3:8b-instruct-q2_K",
        temperature=0, 
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
    normal_transcript = normal_csv['text'].tolist()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=10,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    chunk_data = []
    idx = 0

    # 보이스피싱 데이터 chunking
    for text in scam_transcript:
        chunks = text_splitter.create_documents([text])
        for chunk in chunks:
            chunk_data.append({
                'id': idx,
                'is_scam': 1,
                'text': chunk.page_content
            })
            idx += 1

    # 정상 데이터 chunking
    for text in normal_transcript:
        chunks = text_splitter.create_documents([text])
        for chunk in chunks:
            chunk_data.append({
                'id': idx,
                'is_scam': 0,
                'text': chunk.page_content
            })
            idx += 1

    return pd.DataFrame(chunk_data)

def print_result(chunk_df, num):
    text = chunk_df['text'][num]
    true_label = chunk_df['is_scam'][num]

    result = qa_chain.invoke({"query": text})
    content = result["result"]

    try:
        # 마크다운 코드 블록 제거
        content = re.sub(r'```json\s*|\s*```', '', content).strip()

        # 정규식으로 직접 필요한 정보 추출 (JSON 파싱 우회)
        is_scam_match = re.search(r'"is_scam":\s*"([^"]*)"', content)
        reason_match = re.search(r'"reason":\s*"([^"]*)"', content, re.DOTALL)

        if is_scam_match:
            is_scam_text = is_scam_match.group(1)
            pred = 1 if is_scam_text == "Phishing" else 0
        else:
            pred = 1 if "Phishing" in content and "Normal" not in content else 0

        if reason_match:
            reason = reason_match.group(1)
            # 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
            reason = reason.replace('\\n', '\n').replace('\n            ', '\n').strip()
        else:
            reason = "이유 추출 실패"

    except Exception as error:
        print(f"파싱 오류: {error}")
        print(f"응답 내용 (첫 200자): {content[:200]}...")
        pred = 1 if "Phishing" in content and "Normal" not in content else 0
        reason = "파싱 실패"

    print(f"케이스 {num}: 실제 라벨: {'Phishing' if true_label == 1 else 'Normal'}, "
          f"LLM 예측: {'Phishing' if pred == 1 else 'Normal'}")
    print(f"input text: {text}")
    print(f"판단 이유: {reason}\n")

    return true_label, pred

size = args.size
qa_chain = set_chain()  # 체인
chunk_df = load_test_data(size)

true_labels = []  # 실제 라벨
pred_labels = []  # 예측 라벨

start = pd.Timestamp.now()

for i in range(len(chunk_df)):
    true_label, pred = print_result(chunk_df, i)
    true_labels.append(true_label)
    pred_labels.append(pred)

log_file = open("result.txt", "a")

accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n전체 정확도: {accuracy:.3f}")
end = pd.Timestamp.now()
now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
log_text = f"Input 청크 크기: {size}, 정확도: {accuracy:.3f}, 추론 속도(대사: {len(chunk_df)}): {(end - start).total_seconds():.2f}초, 현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}"
log_file.write(log_text + "\n")

log_file.close()
