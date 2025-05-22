import pandas as pd
import os

def create_test_dataset():
    # 입력 파일 경로
    scam_file = "merged_voice_phishing_dataset.csv"
    normal_file = "normal_conversation_dataset.csv"
    
    # 출력 파일 경로
    output_file = "test_dataset.csv"
    
    # 파일 존재 여부 확인
    if not os.path.exists(scam_file):
        print(f"오류: {scam_file} 파일을 찾을 수 없습니다.")
        return
    if not os.path.exists(normal_file):
        print(f"오류: {normal_file} 파일을 찾을 수 없습니다.")
        return
    
    try:
        # 데이터 읽기
        print("데이터 파일 읽는 중...")
        scam_data = pd.read_csv(scam_file)
        normal_data = pd.read_csv(normal_file)
        
        # 각 데이터셋에서 100개씩 랜덤 샘플링
        print("데이터 샘플링 중...")
        scam_samples = scam_data.sample(n=100, random_state=42)
        normal_samples = normal_data.sample(n=100, random_state=42)
        
        # 데이터 합치기
        print("데이터 병합 중...")
        test_dataset = pd.concat([scam_samples, normal_samples], ignore_index=True)
        
        # 데이터 섞기
        test_dataset = test_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 결과 저장
        print("결과 저장 중...")
        test_dataset.to_csv(output_file, index=False, encoding='utf-8')
        
        # 결과 출력
        print(f"\n생성된 테스트 데이터셋 정보:")
        print(f"전체 샘플 수: {len(test_dataset)}")
        print(f"보이스피싱 샘플 수: {len(test_dataset[test_dataset['is_scam'] == 1])}")
        print(f"일반 통화 샘플 수: {len(test_dataset[test_dataset['is_scam'] == 0])}")
        print(f"\n결과가 {output_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    create_test_dataset()
