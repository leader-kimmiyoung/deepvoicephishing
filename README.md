# 📁 프로젝트: deepvoicephishing

## 🔍 TASK1: Deep voice classification
``` 
deepvoicephishing/
├─ task1/
│  ├─ data_augmentation.py
│  ├─ train.py
│  └─ test.py
```

📌 1. data_augmentation.py

> You need a dataset with the following configuration

> 다음과 같은 구성의 데이터 셋이 필요

``` 
task1/
├─ 0/
│  └─ original/
│      ├─ 1.m4a
│      ├─ 2.m4a 
│      ├─ ...
├─ 1/
│  └─ original/
│      ├─ 1.m4a
│      ├─ 2.m4a 
│      ├─ ...
```

```
make_reverse: original m4a를 역재생하여 m4a로 저장
m4a_to_wav: m4a를 10초씩 split 하여 wav로 저장
random_split: m4a를 랜덤한 시작점에서 10초씩 split하여 wav로 저장
```


📌 2. train.py

>  Save the model to ‘./fine_tuned_model’ using wav2vec2.0

> wav2vec2.0을 사용하여 모델을 './fine_tuned_model'에 저장 

📌 3. test.py

>  Test the saved ‘./fine_tuned_model’ (the sample data uses m4a)

> 저장된 './fine_tuned_model'을 test (sample 데이터는 m4a 사용)

## 🔍 RAG 기반 보이스피싱 탐지
``` 
deepvoicephishing/
├─ RAG/
│  ├─ data_processing.py
│  ├─ vector_store.py
│  ├─ generate_testdata.py
│  ├─ testing_rag.py
│  └─ requirements.env
```

📌 0. requirements.env

> Setting Open AI and Pinecone API key

> Open AI와 Pinecone API key 설정

``` py
OPENAI_API_KEY='input Open AI api key'
PINECONE_API_KEY='input Pinecone api key'
PINECONE_ENV=us-east1-gcp
```

📌 1. data_processing.py

> Split the KorCCVi_v2_cleaned.csv file into chunks based on the specified chunk size

> 설정한 chunk size에 따라 KorCCVi_v2_cleaned.csv 파일을 청크 단위로 분할

```py
python data_processing.py --size [chunk size]
```

📌 2. vector_store.py

>  Creates a vector store based on the data chunks from step 1(The chunk size used here must be the same as in step 1)

> 1에서 분할한 데이터를 기반으로 벡터 스토어를 생성(이때 사용되는 chunk size는 1과 동일해야 함)

```py
python vector_store.py --size [chunk size]
```

📌 3. generate_testdata.py

>  Generate 100 dialogues each for Normal and Scam categories for test dataset

> test 데이터를 생성. Normal과 Scam 각각 100개의 대사 생성

```py
python generate_testdata.py 
```

📌 4. testing_rag.py

>   Chunking and testing of the test data generated in step 3 using the specified chunk size(The chunk size here can differ from those used in steps 1 and 2)

> 3에서 생성한 테스트 데이터를 설정한 chunk size로 분할 및 테스트(이때의 청크 크기는 1, 2와 달라도 무관)

```py
python testing_rag.py --size [chunk size]
```
