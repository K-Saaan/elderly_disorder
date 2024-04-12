# 데이터 정제 및 시각화 처리를 위한 유틸리티

## 폴더 구조

```
📦util
 ┣ 📜README.md
 ┣ 📜utils.py
 ┗ 📜viz.py
 ```

 ## utils.py

 데이터 전처리, 데이터 증강 등 여러 작업 수행에 사용할 수 있는 유틸리티 함수 모음

 ### Features

 - **is_hanja**: 주어진 단어에 한자가 포함되어 있는지 확인
 - **augment_dictionary_with_similar_words**: 주어진 단어 사전의 각 질환 토큰에 대해 유사한 단어를 찾아 사전을 확장

## viz.py

단어 사전에 포함되어 있는 벡터 시각화 관련 함수 모음

### Features

- **tsne_plot**: 단어 사전 내에 있는 단어 벡터 시각화
- **tsne_plot_similar_words**: 주어진 단어와 유사한 단어 라벨링 시각화