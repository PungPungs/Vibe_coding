"""
컨텐츠 기반 필터링 (Content-based Filtering) 모듈
- 상품의 특성(카테고리, 태그, 설명 등)을 분석하여 유사 상품 추천
- TF-IDF: 텍스트 기반 특성 추출
- Feature-based: 카테고리, 가격대 등 구조화된 특성 활용
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_models import Product, User


# ============================================================================
# TF-IDF 기반 컨텐츠 필터링
# ============================================================================

class TFIDFContentFilter:
    """
    TF-IDF 기반 컨텐츠 필터링
    - 상품의 텍스트 정보(이름, 설명, 태그)를 TF-IDF 벡터로 변환
    - 코사인 유사도로 유사 상품 탐색

    TF-IDF (Term Frequency - Inverse Document Frequency):
    - TF: 단어의 문서 내 빈도
    - IDF: 단어의 희소성 (많은 문서에 등장하면 중요도 낮음)
    - TF-IDF = TF * IDF (높을수록 해당 문서의 특징적인 단어)
    """

    def __init__(self, products: Dict[int, Product]):
        """
        초기화 및 TF-IDF 벡터 생성

        Args:
            products (Dict[int, Product]): 상품 ID를 키로 하는 상품 딕셔너리
        """
        self.products = products
        self.product_ids = sorted(products.keys())

        # 각 상품의 텍스트 정보 결합
        # "상품명 설명 태그1 태그2 ... 카테고리"
        product_texts = []
        for pid in self.product_ids:
            product = products[pid]
            text = f"{product.name} {product.description} {' '.join(product.tags)} {product.category}"
            product_texts.append(text)

        # TF-IDF 벡터라이저 생성
        # max_features: 최대 특성 수 (상위 빈도 단어만 사용)
        # ngram_range: (1,2) = unigram + bigram
        # min_df: 최소 문서 빈도 (너무 희귀한 단어 제외)
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            stop_words=None  # 한국어는 별도 불용어 처리 필요
        )

        # TF-IDF 매트릭스 생성
        # shape: (num_products x num_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(product_texts).toarray()

        # 상품 간 유사도 매트릭스 계산
        # shape: (num_products x num_products)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def get_similar_products(
        self,
        product_id: int,
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """
        특정 상품과 유사한 상품 찾기

        Args:
            product_id (int): 타겟 상품 ID
            n_similar (int): 반환할 유사 상품 개수

        Returns:
            List[Tuple[int, float]]: (상품 ID, 유사도 점수) 리스트
        """
        if product_id not in self.product_ids:
            return []

        # 상품 인덱스 찾기
        product_idx = self.product_ids.index(product_id)

        # 유사도 가져오기
        similarities = self.similarity_matrix[product_idx]

        # 자기 자신 제외하고 상위 n개 선택
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]

        similar_products = [
            (self.product_ids[idx], similarities[idx])
            for idx in similar_indices
        ]

        return similar_products

    def recommend_for_user(
        self,
        user: User,
        rating_matrix: np.ndarray,
        user_idx: int,
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        사용자의 과거 선호 상품을 기반으로 유사 상품 추천

        추천 프로세스:
        1. 사용자가 높게 평가한 상품들 찾기
        2. 각 상품과 유사한 상품들 찾기
        3. 유사도 점수 집계 및 정규화
        4. 상위 n개 상품 추천

        Args:
            user (User): 타겟 사용자
            rating_matrix (np.ndarray): User-Item 평점 매트릭스
            user_idx (int): 사용자 인덱스
            n_recommendations (int): 추천할 상품 개수

        Returns:
            List[Tuple[int, float]]: (상품 ID, 추천 점수) 리스트
        """
        user_ratings = rating_matrix[user_idx]

        # 사용자가 높게 평가한 상품 찾기 (평점 4 이상 또는 상위 30%)
        threshold = max(4.0, np.percentile(user_ratings[user_ratings > 0], 70)) if (user_ratings > 0).any() else 4.0
        liked_product_indices = np.where(user_ratings >= threshold)[0]

        # 추천 점수 집계
        recommendation_scores = np.zeros(len(self.product_ids))

        for liked_idx in liked_product_indices:
            # 해당 상품의 유사도 가져오기
            similarities = self.similarity_matrix[liked_idx]

            # 사용자 평점으로 가중치 부여
            weight = user_ratings[liked_idx]
            recommendation_scores += similarities * weight

        # 이미 평가한 상품 제외
        recommendation_scores[user_ratings > 0] = 0

        # 상위 n개 선택
        top_indices = np.argsort(recommendation_scores)[::-1][:n_recommendations]

        recommendations = [
            (self.product_ids[idx], recommendation_scores[idx])
            for idx in top_indices
            if recommendation_scores[idx] > 0
        ]

        return recommendations


# ============================================================================
# Feature 기반 컨텐츠 필터링
# ============================================================================

class FeatureBasedFilter:
    """
    구조화된 특성 기반 필터링
    - 카테고리, 가격대, 브랜드 등 명시적 특성 활용
    - 사용자 프로필과 상품 특성 매칭
    """

    def __init__(self, products: Dict[int, Product], users: Dict[int, User]):
        """
        초기화

        Args:
            products (Dict[int, Product]): 상품 딕셔너리
            users (Dict[int, User]): 사용자 딕셔너리
        """
        self.products = products
        self.users = users
        self.product_ids = sorted(products.keys())

        # 카테고리 인코딩
        self.categories = list(set(p.category for p in products.values()))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # 가격대 구간 설정 (5개 구간)
        all_prices = [p.price for p in products.values()]
        self.price_bins = np.percentile(all_prices, [0, 20, 40, 60, 80, 100])

    def create_product_feature_vector(self, product: Product) -> np.ndarray:
        """
        상품의 특성 벡터 생성

        특성:
        - 카테고리 원핫 인코딩
        - 가격대 (정규화)
        - 평균 평점 (정규화)

        Args:
            product (Product): 상품 객체

        Returns:
            np.ndarray: 특성 벡터
        """
        # 카테고리 원핫 인코딩
        category_vector = np.zeros(len(self.categories))
        if product.category in self.category_to_idx:
            category_vector[self.category_to_idx[product.category]] = 1

        # 가격대 인덱스 (0-4)
        price_level = np.digitize(product.price, self.price_bins) - 1
        price_level = max(0, min(4, price_level))  # 0-4 범위로 제한
        price_vector = np.zeros(5)
        price_vector[price_level] = 1

        # 평균 평점 (0-5 스케일)
        rating = product.features.get('rating_avg', 3.5) / 5.0

        # 모든 특성 결합
        feature_vector = np.concatenate([
            category_vector,
            price_vector,
            [rating]
        ])

        return feature_vector

    def create_user_preference_vector(
        self,
        user: User,
        rating_matrix: np.ndarray,
        user_idx: int
    ) -> np.ndarray:
        """
        사용자의 선호도 벡터 생성
        - 사용자가 높게 평가한 상품들의 특성 벡터 평균

        Args:
            user (User): 사용자 객체
            rating_matrix (np.ndarray): 평점 매트릭스
            user_idx (int): 사용자 인덱스

        Returns:
            np.ndarray: 선호도 벡터
        """
        user_ratings = rating_matrix[user_idx]

        # 높게 평가한 상품들 찾기
        threshold = 4.0
        liked_indices = np.where(user_ratings >= threshold)[0]

        if len(liked_indices) == 0:
            # 평가한 상품이 없으면 선호 카테고리 기반 벡터 생성
            pref_vector = np.zeros(len(self.categories) + 5 + 1)
            for pref_cat in user.preferences:
                if pref_cat in self.category_to_idx:
                    pref_vector[self.category_to_idx[pref_cat]] = 1
            return pref_vector

        # 좋아하는 상품들의 특성 벡터 평균
        feature_vectors = []
        for idx in liked_indices:
            product_id = self.product_ids[idx]
            product = self.products[product_id]
            feature_vectors.append(self.create_product_feature_vector(product))

        preference_vector = np.mean(feature_vectors, axis=0)
        return preference_vector

    def recommend(
        self,
        user: User,
        rating_matrix: np.ndarray,
        user_idx: int,
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        사용자에게 상품 추천

        Args:
            user (User): 타겟 사용자
            rating_matrix (np.ndarray): 평점 매트릭스
            user_idx (int): 사용자 인덱스
            n_recommendations (int): 추천할 상품 개수

        Returns:
            List[Tuple[int, float]]: (상품 ID, 유사도 점수) 리스트
        """
        # 사용자 선호도 벡터 생성
        user_pref_vector = self.create_user_preference_vector(user, rating_matrix, user_idx)

        # 모든 상품에 대해 유사도 계산
        scores = []
        for product_id in self.product_ids:
            product = self.products[product_id]
            product_vector = self.create_product_feature_vector(product)

            # 코사인 유사도 계산
            similarity = np.dot(user_pref_vector, product_vector) / (
                np.linalg.norm(user_pref_vector) * np.linalg.norm(product_vector) + 1e-10
            )
            scores.append(similarity)

        scores = np.array(scores)

        # 이미 평가한 상품 제외
        user_ratings = rating_matrix[user_idx]
        scores[user_ratings > 0] = 0

        # 상위 n개 선택
        top_indices = np.argsort(scores)[::-1][:n_recommendations]

        recommendations = [
            (self.product_ids[idx], scores[idx])
            for idx in top_indices
            if scores[idx] > 0
        ]

        return recommendations


# ============================================================================
# 통합 컨텐츠 기반 필터
# ============================================================================

class CombinedContentFilter:
    """
    TF-IDF와 Feature 기반 필터를 결합한 통합 컨텐츠 필터
    """

    def __init__(
        self,
        products: Dict[int, Product],
        users: Dict[int, User],
        tfidf_weight: float = 0.6,
        feature_weight: float = 0.4
    ):
        """
        초기화

        Args:
            products (Dict[int, Product]): 상품 딕셔너리
            users (Dict[int, User]): 사용자 딕셔너리
            tfidf_weight (float): TF-IDF 필터 가중치
            feature_weight (float): Feature 필터 가중치
        """
        self.tfidf_filter = TFIDFContentFilter(products)
        self.feature_filter = FeatureBasedFilter(products, users)
        self.tfidf_weight = tfidf_weight
        self.feature_weight = feature_weight

    def recommend(
        self,
        user: User,
        rating_matrix: np.ndarray,
        user_idx: int,
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        두 필터의 결과를 가중 평균하여 추천

        Args:
            user (User): 타겟 사용자
            rating_matrix (np.ndarray): 평점 매트릭스
            user_idx (int): 사용자 인덱스
            n_recommendations (int): 추천할 상품 개수

        Returns:
            List[Tuple[int, float]]: (상품 ID, 통합 점수) 리스트
        """
        # 각 필터에서 추천 받기
        tfidf_recs = self.tfidf_filter.recommend_for_user(
            user, rating_matrix, user_idx, n_recommendations * 2
        )
        feature_recs = self.feature_filter.recommend(
            user, rating_matrix, user_idx, n_recommendations * 2
        )

        # 점수 정규화 및 통합
        combined_scores = {}

        # TF-IDF 추천 점수 추가
        if tfidf_recs:
            max_tfidf = max(score for _, score in tfidf_recs)
            for prod_id, score in tfidf_recs:
                normalized_score = score / max_tfidf if max_tfidf > 0 else 0
                combined_scores[prod_id] = normalized_score * self.tfidf_weight

        # Feature 추천 점수 추가
        if feature_recs:
            max_feature = max(score for _, score in feature_recs)
            for prod_id, score in feature_recs:
                normalized_score = score / max_feature if max_feature > 0 else 0
                if prod_id in combined_scores:
                    combined_scores[prod_id] += normalized_score * self.feature_weight
                else:
                    combined_scores[prod_id] = normalized_score * self.feature_weight

        # 점수 기준 정렬
        sorted_recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]

        return sorted_recommendations
