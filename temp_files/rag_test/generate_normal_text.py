import os
import aiohttp
import asyncio
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

GPT_API_KEY = os.getenv("OPENAI_API_KEY")

# 출력 CSV 파일 경로 설정
output_file = "normal_conversation_dataset.csv"

PROMPT_FOR_LLM = """
당신은 일반적인 통화 상담원입니다. 다음 조건에 맞는 일반적인 통화내용을 생성해주세요:

1. 대화 길이:
   - TTS로 읽었을 때 약 7~8초 분량
   - 한글 기준 약 30자 내외
   - 문장은 2문장으로 구성
   - 안녕하세요는 생략

2. 시나리오 선택 (다음 중 하나):
   a) 금융기관 상담:
      - "예금/적금 상품 안내"
      - "대출 상품 상담"
      - "카드 서비스 이용 안내"
   
   b) 공공기관 상담:
      - "보험금/연금 지급 안내"
      - "세금 납부 안내"
      - "복지 서비스 안내"
   
   c) 배송/택배 상담:
      - "배송 지연 안내"
      - "반품/교환 안내"
      - "배송 예정일 확인"
   
   d) 고객 서비스:
      - "상품 문의 응대"
      - "서비스 이용 안내"
      - "고객 불만 접수"

3. 필수 포함 요소:
   - 정중한 어투
   - 명확한 정보 전달
   - 구체적인 안내 내용
   - 필요한 후속 조치 안내

4. 언어 스타일:
   - 친절하고 전문적인 어조
   - 간단명료한 문장
   - 한국어 존댓말 사용
   - 불필요한 수식어 제외

응답은 다음 JSON 형식으로 제공해주세요:
{
    "text": "생성된 대사"
}
"""

async def generate_response(session, data):
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": PROMPT_FOR_LLM},
            {"role": "user", "content": "일반 통화내용을 생성해주세요."}
        ],
        "max_tokens": 300,
        "temperature": 0.8,
        "n": 1
    }
    try:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers,
                              json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                try:
                    parsed_content = json.loads(content)
                    return {
                        "is_scam": 0,  # 일반 통화이므로 0
                        "text": parsed_content["text"]
                    }
                except json.JSONDecodeError:
                    print(f"JSON 파싱 오류: {content}")
                    return None
            elif response.status == 429:
                retry_after = int(response.headers.get('Retry-After', 1))
                print(f"Rate limit 도달. {retry_after}초 후 재시도...")
                await asyncio.sleep(retry_after)
                return await generate_response(session, data)
            else:
                print(f"Error: {response.status} - {await response.text()}")
                return None
    except Exception as e:
        print(f"Exception during API call: {e}")
        return None

async def generate_dataset(num_samples=1000, batch_size=10):
    tasks = []
    results = []

    async with aiohttp.ClientSession() as session:
        for _ in tqdm(range(num_samples), desc="Generating normal texts"):
            tasks.append(generate_response(session, {}))
            
            if len(tasks) >= batch_size:
                batch_results = await asyncio.gather(*tasks)
                results.extend([r for r in batch_results if r is not None])
                tasks = []
                await asyncio.sleep(1)

        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r is not None])

    # CSV 파일로 저장
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["is_scam", "text"])
        writer.writeheader()
        writer.writerows(results)

    print(f"생성된 샘플 수: {len(results)}")
    print(f"결과가 {output_file}에 저장되었습니다.")

def main():
    print("일반 통화내용 생성을 시작합니다...")
    asyncio.run(generate_dataset(num_samples=100, batch_size=5))

if __name__ == "__main__":
    main()
