import os
import aiohttp
import asyncio
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("requirements(test).env")

GPT_API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT_FOR_NORMAL = """
당신은 실제 고객센터 상담원입니다. 아래 조건에 따라 **자연스럽고 상황에 맞는 통화 대사**를 생성하세요. 단, **표현의 다양성을 확보**하고 **문장이 매번 다르게 구성되도록** 하세요.

1. 대화 길이:
   - 한글 기준 300자 내외 
   - ‘안녕하세요’는 생략하고 본론부터 시작

2. 시나리오 카테고리에서 무작위로 1가지 선택 후, 해당 유형에 맞는 내용 생성:
   a) 금융기관 상담:
      - 예금/적금 상품 추천 또는 금리 안내
      - 대출 조건 또는 신청 방법 안내
      - 카드 서비스 변경사항 또는 혜택 안내

   b) 공공기관 상담:
      - 보험금 또는 연금 지급 일정/방법 안내
      - 세금 납부 기한 또는 방식 안내
      - 복지 서비스 신청 또는 지원 자격 안내

   c) 일반 고객 상담:
      - 상품 관련 문의 응답
      - 서비스 이용 방법 또는 접속 안내
      - 고객 불만 접수 및 처리 절차 안내

3. 포함 요소 (모든 응답에 반드시 포함):
   - 정중하고 친절한 말투
   - 핵심 정보 전달 (예: 일정, 방법, 조치 등)
   - 후속 안내 또는 필요한 행동 유도 (예: 확인 부탁, 추가 문의 등)
   - 너무 형식적이지 않고 실제 통화처럼 들리는 문장

4. 언어 스타일:
   - 한국어 존댓말 사용
   - 전문성과 신뢰감을 주는 말투
   - 문장은 간결하게, 불필요한 수식어 없이

5. 표현의 다양성 확보:
   - 문장 구성, 어휘, 말투가 겹치지 않도록 매번 다르게
   - 특정 표현(예: “확인 부탁드립니다”, “도와드리겠습니다”)이 반복되지 않도록 변형 사용
   - 매끄럽고 자연스러운 대화 흐름 유지

응답은 다음 JSON 형식으로 출력하세요:
{
    "text": "생성된 대사"
}
"""

PROMPT_FOR_SCAM = """
당신은 보이스피싱 범죄자입니다. 아래 조건에 따라 실제 사례 기반의 다양한 보이스피싱 통화 내용을 생성하세요. 단, **반복되는 표현은 피하고 각 문장은 실제 사례처럼 구체적이고 현실적으로 구성하세요.**

1. 대화 길이: 
   - 한글 기준 300자 내외 
   - ‘안녕하세요’는 생략하고 본론부터 시작

2. 시나리오 유형 중 랜덤하게 선택 (다양하게 분포되도록):
   - 금융기관 사칭: 계좌 이상 거래 / 대출 조건 / 미납 연체
   - 기관 사칭: 검찰 / 경찰 / 국세청 / 금융감독원 / 사회보장센터 등
   - 택배 사칭: 송장 오류 / 통관 문제 / 미결제 금액
   - IT 사칭: 카카오톡 / 네이버페이 / 휴대폰 해킹
   - 기타 유형: 보증금 요구 / 수수료 / 개인정보 누출 / 명의도용

3. 포함 요소:
   - 실제로 사용될 법한 구체적 계좌, 금액, 기관명
   - 불안감을 유도하는 표현 (예: 조사, 정지, 압류, 협조 등)
   - 정체 불분명하거나 실존하는 기관을 적절히 혼합
   - 빠른 결정을 유도하는 요청 (예: 지금 바로, 늦으면 불이익 등)

4. 언어 스타일:
   - 정중하지만 위기감을 조성하는 말투
   - 실제 상담원, 수사관, 직원처럼 자연스러운 말투
   - 질문형, 지시형, 설명형 문장을 다양하게 섞기
   - 특정 은행(우리은행, 하나은행, 국민은행, 신한은행 중 하나) 포함

5. 다양성 강화:
   - 각 응답은 주제, 말투, 사용 표현이 겹치지 않도록 구성
   - 실제 보이스피싱 음성 데이터를 참고하여 자연스럽게

응답은 다음 JSON 형식으로 출력하세요:
{
    "text": "생성된 대사"
}
"""

async def generate_response(session, prompt, is_scam):
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "통화 내용을 생성해주세요."}
        ],
        "max_tokens": 300,
        "temperature": 0.8,
        "n": 1
    }
    try:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                try:
                    parsed_content = json.loads(content)
                    return {
                        "is_scam": is_scam,
                        "text": parsed_content["text"]
                    }
                except json.JSONDecodeError:
                    print(f"JSON 파싱 오류: {content}")
                    return None
            elif response.status == 429:
                retry_after = int(response.headers.get('Retry-After', 1))
                print(f"Rate limit 도달. {retry_after}초 후 재시도...")
                await asyncio.sleep(retry_after)
                return await generate_response(session, prompt, is_scam)
            else:
                print(f"Error: {response.status} - {await response.text()}")
                return None
    except Exception as e:
        print(f"Exception during API call: {e}")
        return None

async def generate_dataset(num_samples=100, batch_size=10, is_scam=0, prompt=None, output_file="output.csv"):
    tasks = []
    results = []

    async with aiohttp.ClientSession() as session:
        for _ in tqdm(range(num_samples), desc="Generating texts"):
            tasks.append(generate_response(session, prompt, is_scam))

            if len(tasks) >= batch_size:
                batch_results = await asyncio.gather(*tasks)
                results.extend([r for r in batch_results if r is not None])
                tasks = []
                await asyncio.sleep(1)

        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r is not None])

    # 저장
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["is_scam", "text"])
        writer.writeheader()
        writer.writerows(results)

    print(f"생성된 샘플 수: {len(results)}")
    print(f"결과가 {output_file}에 저장되었습니다.")

output_dir = "./dataset(for test)"
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

print("✅ 일반 통화내용 생성을 시작합니다...")
asyncio.run(generate_dataset(num_samples=100, batch_size=5, is_scam=0, prompt=PROMPT_FOR_NORMAL, output_file="./dataset(for test)/normal.csv"))

print("🚨 보이스피싱 통화내용 생성을 시작합니다...")
asyncio.run(generate_dataset(num_samples=100, batch_size=5, is_scam=1, prompt=PROMPT_FOR_SCAM, output_file="./dataset(for test)/scam.csv"))
