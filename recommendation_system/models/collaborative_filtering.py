"""
협업 필터링 모델
- 사용자 기반 협업 필터링 (User-based CF)
- 아이템 기반 협업 필터링 (Item-based CF)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass


@dataclass
class CFRecommendation:
    """협업 필터링 추천 결과"""
    product_name: str
    predicted_score: float
    similar_users: List[int]
    confidence: float


class CollaborativeFiltering:
    """
    협업 필터링 추천 모델

    주요 기능:
    - 사용자 기반 협업 필터링: 유사한 사용자가 좋아한 상품 추천
    - 아이템 기반 협업 필터링: 유사한 상품 추천
    - 이웃 기반 예측 및 추천
    """

    def __init__(
        self,
        method: str = 'user',
        n_neighbors: int = 20,
        min_common_items: int = 1,
        similarity_threshold: float = 0.1
    ):
        """
        Args:
            method: 협업 필터링 방식 ('user' 또는 'item')
            n_neighbors: 고려할 이웃 수
            min_common_items: 유사도 계산에 필요한 최소 공통 항목 수
            similarity_threshold: 유사도 임계값
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items
        self.similarity_threshold = similarity_threshold

        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_ids = None
        self.product_names = None
        self.user_to_idx = None
        self.product_to_idx = None
        self.idx_to_user = None
        self.idx_to_product = None
        self.user_means = None

    def fit(self, user_item_matrix: pd.DataFrame):
        """
        모델 학습

        Args:
            user_item_matrix: 사용자-상품 상호작용 매트릭스
        """
        self.user_item_matrix = user_item_matrix.values
        self.user_ids = list(user_item_matrix.index)
        self.product_names = list(user_item_matrix.columns)

        # 인덱스 매핑
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.product_to_idx = {pname: idx for idx, pname in enumerate(self.product_names)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.idx_to_product = {idx: pname for pname, idx in self.product_to_idx.items()}

        # 사용자 평균 평점 계산 (0이 아닌 값만)
        self.user_means = np.zeros(len(self.user_ids))
        for i in range(len(self.user_ids)):
            non_zero = self.user_item_matrix[i][self.user_item_matrix[i] > 0]
            self.user_means[i] = non_zero.mean() if len(non_zero) > 0 else 0

        # 유사도 행렬 계산
        if self.method == 'user':
            self._compute_user_similarity()
        else:
            self._compute_item_similarity()

        print(f"Collaborative Filtering fitted: {len(self.user_ids)} users, {len(self.product_names)} products")

    def _compute_user_similarity(self):
        """사용자 유사도 행렬 계산 (코사인 유사도)"""
        # 평균 중심화된 행렬
        centered_matrix = self.user_item_matrix.copy()
        for i in range(len(self.user_ids)):
            mask = centered_matrix[i] > 0
            centered_matrix[i][mask] -= self.user_means[i]

        # 코사인 유사도 계산
        self.user_similarity_matrix = cosine_similarity(centered_matrix)

        # 자기 자신과의 유사도는 0으로 설정
        np.fill_diagonal(self.user_similarity_matrix, 0)

        print(f"User similarity matrix computed: {self.user_similarity_matrix.shape}")

    def _compute_item_similarity(self):
        """아이템 유사도 행렬 계산"""
        # 상품 기준으로 전치
        item_matrix = self.user_item_matrix.T

        # 코사인 유사도 계산
        self.item_similarity_matrix = cosine_similarity(item_matrix)

        # 자기 자신과의 유사도는 0으로 설정
        np.fill_diagonal(self.item_similarity_matrix, 0)

        print(f"Item similarity matrix computed: {self.item_similarity_matrix.shape}")

    def _get_similar_users(self, user_idx: int, n: int = None) -> List[Tuple[int, float]]:
        """
        유사한 사용자 검색

        Args:
            user_idx: 대상 사용자 인덱스
            n: 반환할 유사 사용자 수

        Returns:
            (사용자 인덱스, 유사도) 튜플 리스트
        """
        if n is None:
            n = self.n_neighbors

        similarities = self.user_similarity_matrix[user_idx]

        # 유사도 기준 정렬
        similar_indices = np.argsort(similarities)[::-1]

        # 임계값 이상인 사용자만 선택
        result = []
        for idx in similar_indices:
            if len(result) >= n:
                break
            if similarities[idx] >= self.similarity_threshold:
                result.append((idx, similarities[idx]))

        return result

    def _get_similar_items(self, item_idx: int, n: int = None) -> List[Tuple[int, float]]:
        """
        유사한 상품 검색

        Args:
            item_idx: 대상 상품 인덱스
            n: 반환할 유사 상품 수

        Returns:
            (상품 인덱스, 유사도) 튜플 리스트
        """
        if n is None:
            n = self.n_neighbors

        similarities = self.item_similarity_matrix[item_idx]

        # 유사도 기준 정렬
        similar_indices = np.argsort(similarities)[::-1]

        # 임계값 이상인 상품만 선택
        result = []
        for idx in similar_indices:
            if len(result) >= n:
                break
            if similarities[idx] >= self.similarity_threshold:
                result.append((idx, similarities[idx]))

        return result

    def predict_user_based(
        self,
        user_idx: int,
        item_idx: int
    ) -> Tuple[float, List[int], float]:
        """
        사용자 기반 예측

        Args:
            user_idx: 사용자 인덱스
            item_idx: 상품 인덱스

        Returns:
            (예측 점수, 유사 사용자 리스트, 신뢰도)
        """
        similar_users = self._get_similar_users(user_idx)

        if not similar_users:
            return self.user_means[user_idx], [], 0.0

        numerator = 0.0
        denominator = 0.0
        contributing_users = []

        for neighbor_idx, similarity in similar_users:
            neighbor_rating = self.user_item_matrix[neighbor_idx, item_idx]

            if neighbor_rating > 0:  # 이웃이 해당 상품을 평가한 경우
                # 평균 중심화된 평점 사용
                centered_rating = neighbor_rating - self.user_means[neighbor_idx]
                numerator += similarity * centered_rating
                denominator += abs(similarity)
                contributing_users.append(self.idx_to_user[neighbor_idx])

        if denominator == 0:
            return self.user_means[user_idx], [], 0.0

        predicted = self.user_means[user_idx] + (numerator / denominator)

        # 점수 범위 제한 (1-5)
        predicted = max(1, min(5, predicted))

        # 신뢰도: 기여한 이웃 수 / 전체 이웃 수
        confidence = len(contributing_users) / max(len(similar_users), 1)

        return predicted, contributing_users, confidence

    def predict_item_based(
        self,
        user_idx: int,
        item_idx: int
    ) -> Tuple[float, List[str], float]:
        """
        아이템 기반 예측

        Args:
            user_idx: 사용자 인덱스
            item_idx: 상품 인덱스

        Returns:
            (예측 점수, 유사 상품 리스트, 신뢰도)
        """
        similar_items = self._get_similar_items(item_idx)

        if not similar_items:
            return self.user_means[user_idx], [], 0.0

        numerator = 0.0
        denominator = 0.0
        contributing_items = []

        for neighbor_idx, similarity in similar_items:
            user_rating = self.user_item_matrix[user_idx, neighbor_idx]

            if user_rating > 0:  # 사용자가 유사 상품을 평가한 경우
                numerator += similarity * user_rating
                denominator += similarity
                contributing_items.append(self.idx_to_product[neighbor_idx])

        if denominator == 0:
            return self.user_means[user_idx], [], 0.0

        predicted = numerator / denominator

        # 점수 범위 제한
        predicted = max(1, min(5, predicted))

        # 신뢰도
        confidence = len(contributing_items) / max(len(similar_items), 1)

        return predicted, contributing_items, confidence

    def predict(self, user_id: int, product_name: str) -> Tuple[float, float]:
        """
        평점 예측

        Args:
            user_id: 사용자 ID
            product_name: 상품명

        Returns:
            (예측 점수, 신뢰도)
        """
        if user_id not in self.user_to_idx:
            return 3.0, 0.0  # 알 수 없는 사용자

        if product_name not in self.product_to_idx:
            return 3.0, 0.0  # 알 수 없는 상품

        user_idx = self.user_to_idx[user_id]
        item_idx = self.product_to_idx[product_name]

        # 이미 평가한 상품인 경우
        if self.user_item_matrix[user_idx, item_idx] > 0:
            return self.user_item_matrix[user_idx, item_idx], 1.0

        if self.method == 'user':
            predicted, _, confidence = self.predict_user_based(user_idx, item_idx)
        else:
            predicted, _, confidence = self.predict_item_based(user_idx, item_idx)

        return predicted, confidence

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_purchased: bool = True
    ) -> List[CFRecommendation]:
        """
        사용자에게 상품 추천

        Args:
            user_id: 사용자 ID
            top_n: 추천 상품 수
            exclude_purchased: 이미 구매한 상품 제외 여부

        Returns:
            추천 결과 리스트
        """
        if user_id not in self.user_to_idx:
            # 콜드 스타트: 인기 상품 추천
            return self._recommend_popular(top_n)

        user_idx = self.user_to_idx[user_id]
        predictions = []

        for item_idx, product_name in enumerate(self.product_names):
            # 이미 구매한 상품 제외
            if exclude_purchased and self.user_item_matrix[user_idx, item_idx] > 0:
                continue

            if self.method == 'user':
                score, similar_users, confidence = self.predict_user_based(user_idx, item_idx)
            else:
                score, _, confidence = self.predict_item_based(user_idx, item_idx)
                similar_users = []

            predictions.append(CFRecommendation(
                product_name=product_name,
                predicted_score=score,
                similar_users=similar_users[:3],  # 상위 3명만
                confidence=confidence
            ))

        # 점수 기준 정렬
        predictions.sort(key=lambda x: (x.predicted_score, x.confidence), reverse=True)

        return predictions[:top_n]

    def _recommend_popular(self, top_n: int) -> List[CFRecommendation]:
        """인기 상품 추천 (콜드 스타트 대응)"""
        # 상품별 평균 평점 계산
        avg_ratings = np.zeros(len(self.product_names))
        for item_idx in range(len(self.product_names)):
            ratings = self.user_item_matrix[:, item_idx]
            non_zero = ratings[ratings > 0]
            avg_ratings[item_idx] = non_zero.mean() if len(non_zero) > 0 else 0

        # 평점 기준 정렬
        sorted_indices = np.argsort(avg_ratings)[::-1]

        recommendations = []
        for idx in sorted_indices[:top_n]:
            recommendations.append(CFRecommendation(
                product_name=self.product_names[idx],
                predicted_score=avg_ratings[idx],
                similar_users=[],
                confidence=0.5  # 인기 기반이므로 중간 신뢰도
            ))

        return recommendations

    def get_similar_products(
        self,
        product_name: str,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        유사 상품 검색

        Args:
            product_name: 상품명
            top_n: 반환할 유사 상품 수

        Returns:
            (상품명, 유사도) 튜플 리스트
        """
        if self.item_similarity_matrix is None:
            self._compute_item_similarity()

        if product_name not in self.product_to_idx:
            return []

        item_idx = self.product_to_idx[product_name]
        similar_items = self._get_similar_items(item_idx, top_n)

        return [
            (self.idx_to_product[idx], sim)
            for idx, sim in similar_items
        ]

    def get_similar_users(
        self,
        user_id: int,
        top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        유사 사용자 검색

        Args:
            user_id: 사용자 ID
            top_n: 반환할 유사 사용자 수

        Returns:
            (사용자 ID, 유사도) 튜플 리스트
        """
        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        similar_users = self._get_similar_users(user_idx, top_n)

        return [
            (self.idx_to_user[idx], sim)
            for idx, sim in similar_users
        ]


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from utils.data_processor import DataProcessor, create_sample_dataframe

    # 데이터 준비
    processor = DataProcessor()
    df = create_sample_dataframe()
    processed_df = processor.process_data(df)
    matrix, user_map, product_map = processor.create_user_item_matrix()

    # 협업 필터링 모델 학습
    cf_model = CollaborativeFiltering(method='user', n_neighbors=3)
    cf_model.fit(matrix)

    # 추천
    for user_id in list(processor.user_profiles.keys())[:2]:
        print(f"\n=== User {user_id} 추천 ===")
        recommendations = cf_model.recommend(user_id, top_n=3)
        for rec in recommendations:
            print(f"  {rec.product_name[:30]}... (점수: {rec.predicted_score:.2f}, 신뢰도: {rec.confidence:.2f})")

    # 유사 상품
    sample_product = list(processor.product_info.keys())[0]
    print(f"\n=== '{sample_product[:30]}...'와 유사한 상품 ===")
    similar = cf_model.get_similar_products(sample_product, top_n=3)
    for name, sim in similar:
        print(f"  {name[:30]}... (유사도: {sim:.3f})")
