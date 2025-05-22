import os
import pickle
import random
import json
import re
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics import accuracy_score


# .env 파일에서 API 키 로드
load_dotenv(find_dotenv())

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
        index_name="scam-pinecone",
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
        base_url="http://localhost:11434",
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
def load_test_data(file_path="test_dataset.csv"):
    try:
        # CSV 파일 로드
        df = pd.read_csv(file_path)
        print(f"테스트 데이터 로드 완료: {len(df)}개 샘플")
        print(f"보이스피싱 샘플: {len(df[df['is_scam'] == 1])}개")
        print(f"일반 통화 샘플: {len(df[df['is_scam'] == 0])}개")
        return df
    except Exception as e:
        print(f"테스트 데이터 로드 중 오류 발생: {e}")
        return None


def main():
    qa_chain = set_chain()  # 체인
    
    # 테스트 데이터 로드
    test_data = load_test_data()
    if test_data is None:
        print("테스트 데이터를 로드할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    true_labels = []  # 실제 라벨
    pred_labels = []  # 예측 라벨
    
    # 테스트 데이터 준비
    all_samples = list(zip(test_data['text'], test_data['is_scam']))
    
    # 테스트 케이스 셔플
    random.shuffle(all_samples)
    
    print("테스트 진행 중...")
    for i, (text, true_label) in enumerate(all_samples, 1):
        try:
            # LLM 호출
            result = qa_chain({"query": text})
            content = result["result"]
            
            # 언어모델 응답 파싱
            try:
                # 마크다운 코드 블록 제거
                content = re.sub(r'```json\s*|\s*```', '', content)
                
                # JSON 문자열 추출
                json_str = content.strip()
                
                # JSON 파싱
                parsed_data = json.loads(json_str)
                segments = parsed_data.get('segments', [])
                
                if segments:
                    segment = segments[0]
                    is_scam_text = segment.get('is_scam', '')
                    reason = segment.get('reason', '이유 없음')
                    pred = 1 if is_scam_text == "보이스피싱" else 0
                else:
                    pred = 0
                    reason = "segments 정보 없음"
                    
            except Exception as json_error:
                print(f"JSON 파싱 오류: {json_error}")
                print(f"응답 내용: {content}")
                # JSON 파싱 실패 시 텍스트 기반 판단
                pred = 1 if "보이스피싱" in content and "정상" not in content else 0
                reason = "JSON 파싱 실패"
            
            true_labels.append(true_label)
            pred_labels.append(pred)
            
            print(f"케이스 {i}: 실제 라벨: {'보이스피싱' if true_label == 1 else '정상'}, "
                  f"LLM 예측: {'보이스피싱' if pred == 1 else '정상'}")
            print(f"판단 이유: {reason}\n")
        
        except Exception as e:
            print(f"케이스 {i} 처리 중 오류 발생: {e}")
            continue
    
    # 정확도 계산
    if true_labels:
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\n전체 정확도: {accuracy:.3f}")
    else:
        print("\n처리된 케이스가 없어 정확도를 계산할 수 없습니다.")


if __name__ == "__main__":
    main()