FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn==0.27.0

COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p temp calls logs

# 포트 설정
EXPOSE 5000

# 서버 실행
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"] 