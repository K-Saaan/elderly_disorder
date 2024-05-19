# 폴더별 설명

## /Training & /Validation
- ※ Training & Validation은 감성대화말뭉치 가공 전 데이터
- 라벨링 데이터/감성대화말뭉치(최종데이터)_Training.json: 레이블링 O
- 원천 데이터/감성대화말뭉치(최종데이터)_Training.xlsx: 레이블링 X
- 라벨링 데이터/감성대화말뭉치(최종데이터)_Validation.json: 레이블링 O
- 원천 데이터/감성대화말뭉치(최종데이터)_Validation.xlsx: 레이블링 X

## /train
- 감성대화말뭉치 데이터셋의 json을 only csv형태로 바꾼 것 => train_df
- 감성대화말뭉치 데이터셋의 노년층 데이터 + 노년층대상_감성분류모델 데이터셋 => train

## /노년층대상_감성분류모델
- 감성대화말뭉치 외 데이터셋
- 노년층대상_감성분류모델/augmentation/aug_df.csv => 데이터셋 가공한 형태
- aug_df는 `240411 현재 형태소 분석 증강을 위한 데이터

## *.csv
- train.csv: 정신질환 분류 데이터셋
- vocab_augmented.csv: 증강 된 진단기준 단어사전