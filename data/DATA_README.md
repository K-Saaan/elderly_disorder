# 폴더별 설명

## /Training
- 라벨링 데이터/감성대화말뭉치(최종데이터)_Training.json: 레이블링 O
- 원천 데이터/감성대화말뭉치(최종데이터)_Training.xlsx: 레이블링 X

## /Validation
- 라벨링 데이터/감성대화말뭉치(최종데이터)_Validation.json: 레이블링 O
- 원천 데이터/감성대화말뭉치(최종데이터)_Validation.xlsx: 레이블링 X

## ※ Training & Validation은 감성대화말뭉치 가공 전 데이터
- 즉, train_df 쓰면 됨

## /train
- 감성대화말뭉치 데이터셋의 json을 only csv형태로 바꾼 것

## /노년층대상_감성분류모델
- 감성대화말뭉치 외 데이터셋
- 노년층대상_감성분류모델/augmentation/aug_df.csv => 데이터셋 가공한 형태