from kiwipiepy import Kiwi
import re

hanja_range = re.compile(r'[\u4e00-\u9fff]+')

def is_hanja(word: str) -> bool:
    """
    주어진 단어에 한자가 포함되어 있는지 확인

    Args:
        word (str): 확인할 단어

    Returns:
        bool: 단어에 한자가 포함되어 있다면 True, 아니면 False
    """
    return bool(hanja_range.search(word))


def augment_dictionary_with_similar_words(model, word_dictionary: dict, top_n: int=3, similarity_threshold: float=0.7) -> tuple:
    """
    주어진 단어 사전의 각 질환 토큰에 대해 유사한 단어를 찾아 사전을 확장

    Args:
        model (Word2Vec model): 단어 간 유사도를 계산할 Word2Vec 모델
        word_dictionary (dict): 각 정신 질환과 관련된 토큰들의 사전
                                "mental_disorder"와 "tokens" 키를 가짐
        top_n (int, optional): 각 토큰에 대해 가져올 유사 단어의 최대 개수. Defaults to 3.
        similarity_threshold (float, optional): 단어를 추가하기 위한 유사도 임계값. Defaults to 0.7.

    Returns:
        tuple: 확장된 단어 사전, 새로 추가된 토큰들, 각 토큰에 추가된 유사한 단어들의 사전으로 구성된 튜플
    """
    kiwi = Kiwi()

    augmented_dictionary = {}
    new_tokens_added = {}
    token_similar_words_added = {}
    analyzed_cache = {} # 형태소 분석 결과 캐싱

    target_tags = {'NNG', 'NNP', 'NR', 'NP', 'VCN', 'MAG', 'XPN'}

    for disorder, tokens in zip(word_dictionary["mental_disorder"], word_dictionary["tokens"]):
        augmented_tokens_set = set(tokens)
        new_tokens_for_disorder = []
        token_similar_words = {}

        for token in tokens:
            try:
                if token in analyzed_cache:
                    similar_words = analyzed_cache[token]
                else:
                    similar_words = model.most_similar(token, topn=top_n)
                    analyzed_cache[token] = similar_words

                similar_words_added = []

                for word, similarity in similar_words:
                    if similarity < similarity_threshold or is_hanja(word) or word in augmented_tokens_set:
                        continue

                    analyzed = kiwi.analyze(word)[0][0]
                    word_tags = {morph.tag for morph in analyzed} # 형태소 태그를 집합으로 변환

                    if word_tags & target_tags: # 교집합이 존재하는 경우
                        augmented_tokens_set.add(word)
                        new_tokens_for_disorder.append(word)
                        similar_words_added.append(word)

                if similar_words_added:
                    token_similar_words[token] = similar_words_added
            
            except KeyError:
                continue

        augmented_dictionary[disorder] = list(augmented_tokens_set) # 리스트로 변환
        new_tokens_added[disorder] = new_tokens_for_disorder
        token_similar_words_added[disorder] = token_similar_words

    return augmented_dictionary, new_tokens_added, token_similar_words_added