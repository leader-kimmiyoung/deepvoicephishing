# 선행 연구자가 만들어 놓은 금감원 보이스피싱 데이터(csv 형식) 사용
# 데이터 전처리 코드

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from dotenv import load_dotenv 
import os 
import argparse

parser = argparse.ArgumentParser(description="스크립트 설명")
parser.add_argument('--size', type=int, default=256, help='chunk의 크기') 
args = parser.parse_args()

# 환경 변수 불러오기
load_dotenv("requirements.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# csv 데이터 불러오기
df = pd.read_csv("./KorCCVi_v2_cleaned.csv")

# 데이터 확인
print(df.head())
print("-"*100)

# 보이스피싱 통화(label=1) 데이터 추출
scam_df = df[df['label'] == 1]

# transcript(문자열) 열 추출
transcript = scam_df['transcript'].tolist()

# 데이터 확인
for i in range(3):
    print(transcript[i])
    print("-"*100)
    
# 청크 저장
def save_chunks(chunk_size, output_dir="chunk_data"):
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=10,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    # 문자열 분할
    chunks = text_splitter.create_documents(transcript)

    # 데이터 확인
    for i in range(3):
        print(chunks[i])
        print("-"*100)
        
    # 저장 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 청크를 DataFrame으로 변환
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            'id': i,
            'is_scam': 1,  # 보이스피싱 데이터이므로 1로 설정
            'text': chunk.page_content
        })
    
    # DataFrame 생성
    chunk_df = pd.DataFrame(chunk_data)
    
    # CSV로 저장
    csv_path = os.path.join(output_dir, f"chunks({chunk_size}).csv")
    chunk_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"청크가 {csv_path}에 저장되었습니다.")
    print(f"총 청크 수: {len(chunks)}")

save_chunks(args.size)
