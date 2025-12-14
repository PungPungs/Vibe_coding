"""
하이브리드 추천 시스템
- 협업 필터링과 컨텐츠 기반 필터링을 결합
- 각 방법의 장점을 활용하고 단점을 보완
- 다양한 결합 전략 제공
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_models import Dataset, User
from models.collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization
from models.content_based_filtering import CombinedContentFilter


# ============================================================================
# 하이브리드 전략 Enum
# ============================================================================

class HybridStrategy:
    """하이브리드 결합 전략"""
    WEIGHTED = "weighted"           # 가중 평균
    SWITCHING = "switching"         # 상황에 따라 전환
    MIXED = "mixed"                 # 혼합 (각 방법에서 일부씩)
    FEATURE_COMBINATION = "feature" # 특성 결합


# ============================================================================
# 가중 평균 하이브리드 추천기
# ============================================================================

class WeightedHybridRecommender:
    """
    가중 평균 하이브리드 추천기
    - 여러 추천 알고리즘의 결과를 가중치를 두어 결합
    - 가장 간단하고 효과적인 하이브리드 방식

    장점:
    - 구현이 간단함
    - 각 알고리즘의 장점을 모두 활용
    - 가중치 조정으로 성능 최적화 가능

    단점:
    - 최적의 가중치 찾기가 어려움
    - 모든 알고리즘을 실행해야 하므로 계산 비용 증가
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Dict[str, float] = None
    ):
        """
        초기화

        Args:
            dataset (Dataset): 데이터셋
            weights (Dict[str, float]): 각 알고리즘의 가중치
                {
                    'user_cf': 0.25,
                    'item_cf': 0.25,
                    'mf': 0.25,
                    'content': 0.25
                }
        """
        self.dataset = dataset

        # 기본 가중치 설정
        if weights is None:
            self.weights = {
                'user_cf': 0.2,   # User-based CF
                'item_cf': 0.2,   # Item-based CF
                'mf': 0.3,        # Matrix Factorization (일반적으로 성능 우수)
                'content': 0.3    # Content-based (Cold start 문제 해결)
            }
        else:
            self.weights = weights

        # 각 추천 알고리즘 초기화
        rating_matrix = dataset.build_rating_matrix()

        self.user_cf = UserBasedCF(rating_matrix)
        self.item_cf = ItemBasedCF(rating_matrix)
        self.mf = MatrixFactorization(rating_matrix, n_factors=20)
        self.content_filter = CombinedContentFilter(dataset.products, dataset.users)

    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        하이브리드 추천 수행

        프로세스:
        1. 각 알고리즘에서 추천 결과 받기
        2. 상품별 점수를 정규화
        3. 가중 평균 계산
        4. 최종 상위 n개 상품 반환

        Args:
            user_id (int): 타겟 사용자 ID
            n_recommendations (int): 추천할 상품 개수

        Returns:
            List[Tuple[int, float]]: (상품 ID, 추천 점수) 리스트
        """
        # 사용자 인덱스 찾기
        user_ids = sorted(self.dataset.users.keys())
        if user_id not in user_ids:
            return []

        user_idx = user_ids.index(user_id)
        user = self.dataset.users[user_id]

        # 각 알고리즘에서 추천 받기 (더 많이 받아서 풀 활용)
        n_fetch = n_recommendations * 3

        user_cf_recs = self.user_cf.recommend(user_idx, n_fetch)
        item_cf_recs = self.item_cf.recommend(user_idx, n_fetch)
        mf_recs = self.mf.recommend(user_idx, n_fetch)
        content_recs = self.content_filter.recommend(
            user,
            self.dataset.rating_matrix,
            user_idx,
            n_fetch
        )

        # 상품 ID를 인덱스로 변환
        product_ids = sorted(self.dataset.products.keys())

        # 정규화 및 통합
        combined_scores = {}

        # User-based CF
        if user_cf_recs:
            max_score = max(score for _, score in user_cf_recs) if user_cf_recs else 1.0
            for idx, score in user_cf_recs:
                prod_id = product_ids[idx]
                normalized = score / max_score if max_score > 0 else 0
                combined_scores[prod_id] = normalized * self.weights['user_cf']

        # Item-based CF
        if item_cf_recs:
            max_score = max(score for _, score in item_cf_recs) if item_cf_recs else 1.0
            for idx, score in item_cf_recs:
                prod_id = product_ids[idx]
                normalized = score / max_score if max_score > 0 else 0
                if prod_id in combined_scores:
                    combined_scores[prod_id] += normalized * self.weights['item_cf']
                else:
                    combined_scores[prod_id] = normalized * self.weights['item_cf']

        # Matrix Factorization
        if mf_recs:
            max_score = max(score for _, score in mf_recs) if mf_recs else 1.0
            for idx, score in mf_recs:
                prod_id = product_ids[idx]
                normalized = score / max_score if max_score > 0 else 0
                if prod_id in combined_scores:
                    combined_scores[prod_id] += normalized * self.weights['mf']
                else:
                    combined_scores[prod_id] = normalized * self.weights['mf']

        # Content-based
        if content_recs:
            max_score = max(score for _, score in content_recs) if content_recs else 1.0
            for prod_id, score in content_recs:
                normalized = score / max_score if max_score > 0 else 0
                if prod_id in combined_scores:
                    combined_scores[prod_id] += normalized * self.weights['content']
                else:
                    combined_scores[prod_id] = normalized * self.weights['content']

        # 점수 기준 정렬
        sorted_recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]

        return sorted_recommendations


# ============================================================================
# 스위칭 하이브리드 추천기
# ============================================================================

class SwitchingHybridRecommender:
    """
    스위칭 하이브리드 추천기
    - 상황에 따라 다른 추천 알고리즘 선택
    - 예: Cold start 문제가 있으면 컨텐츠 기반, 아니면 협업 필터링

    장점:
    - 상황에 맞는 최적의 알고리즘 사용
    - 계산 효율적 (하나의 알고리즘만 실행)

    단점:
    - 전환 로직이 복잡할 수 있음
    - 경계 상황에서 성능 불안정
    """

    def __init__(self, dataset: Dataset):
        """
        초기화

        Args:
            dataset (Dataset): 데이터셋
        """
        self.dataset = dataset
        rating_matrix = dataset.build_rating_matrix()

        # 모든 추천기 초기화
        self.user_cf = UserBasedCF(rating_matrix)
        self.item_cf = ItemBasedCF(rating_matrix)
        self.mf = MatrixFactorization(rating_matrix, n_factors=20)
        self.content_filter = CombinedContentFilter(dataset.products, dataset.users)

    def _is_cold_start_user(self, user_id: int, threshold: int = 5) -> bool:
        """
        Cold start 사용자 판별
        - 평가한 상품 수가 threshold 미만이면 cold start

        Args:
            user_id (int): 사용자 ID
            threshold (int): 평가 개수 임계값

        Returns:
            bool: Cold start 여부
        """
        user_ids = sorted(self.dataset.users.keys())
        if user_id not in user_ids:
            return True

        user_idx = user_ids.index(user_id)
        user_ratings = self.dataset.rating_matrix[user_idx]
        num_ratings = np.count_nonzero(user_ratings)

        return num_ratings < threshold

    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        스위칭 전략으로 추천

        전략:
        1. Cold start 사용자 -> Content-based
        2. 평가 많은 사용자 -> Matrix Factorization (최고 성능)
        3. 중간 사용자 -> Item-based CF (안정적)

        Args:
            user_id (int): 타겟 사용자 ID
            n_recommendations (int): 추천할 상품 개수

        Returns:
            List[Tuple[int, float]]: (상품 ID, 추천 점수) 리스트
        """
        user_ids = sorted(self.dataset.users.keys())
        if user_id not in user_ids:
            return []

        user_idx = user_ids.index(user_id)
        user = self.dataset.users[user_id]

        # 사용자 평가 수 확인
        user_ratings = self.dataset.rating_matrix[user_idx]
        num_ratings = np.count_nonzero(user_ratings)

        product_ids = sorted(self.dataset.products.keys())

        # 스위칭 로직
        if num_ratings < 5:
            # Cold start: Content-based 사용
            print(f"[Switching] Cold start 사용자 -> Content-based 사용")
            recommendations = self.content_filter.recommend(
                user,
                self.dataset.rating_matrix,
                user_idx,
                n_recommendations
            )
        elif num_ratings >= 20:
            # 활발한 사용자: Matrix Factorization 사용
            print(f"[Switching] 활발한 사용자 -> Matrix Factorization 사용")
            recs = self.mf.recommend(user_idx, n_recommendations)
            recommendations = [(product_ids[idx], score) for idx, score in recs]
        else:
            # 중간 사용자: Item-based CF 사용
            print(f"[Switching] 중간 사용자 -> Item-based CF 사용")
            recs = self.item_cf.recommend(user_idx, n_recommendations)
            recommendations = [(product_ids[idx], score) for idx, score in recs]

        return recommendations


# ============================================================================
# 혼합(Mixed) 하이브리드 추천기
# ============================================================================

class MixedHybridRecommender:
    """
    혼합 하이브리드 추천기
    - 여러 추천기에서 일부씩 추천받아 섞음
    - 다양성(Diversity) 향상

    장점:
    - 추천 결과의 다양성 증가
    - 각 알고리즘의 강점 활용

    단점:
    - 추천 품질이 일관되지 않을 수 있음
    - 정확도가 다소 낮을 수 있음
    """

    def __init__(self, dataset: Dataset):
        """
        초기화

        Args:
            dataset (Dataset): 데이터셋
        """
        self.dataset = dataset
        rating_matrix = dataset.build_rating_matrix()

        self.mf = MatrixFactorization(rating_matrix, n_factors=20)
        self.item_cf = ItemBasedCF(rating_matrix)
        self.content_filter = CombinedContentFilter(dataset.products, dataset.users)

    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        distribution: Dict[str, float] = None
    ) -> List[Tuple[int, float]]:
        """
        혼합 전략으로 추천

        Args:
            user_id (int): 타겟 사용자 ID
            n_recommendations (int): 추천할 상품 개수
            distribution (Dict[str, float]): 각 알고리즘의 추천 비율
                예: {'mf': 0.5, 'item_cf': 0.3, 'content': 0.2}

        Returns:
            List[Tuple[int, float]]: (상품 ID, 추천 점수) 리스트
        """
        if distribution is None:
            distribution = {'mf': 0.5, 'item_cf': 0.3, 'content': 0.2}

        user_ids = sorted(self.dataset.users.keys())
        if user_id not in user_ids:
            return []

        user_idx = user_ids.index(user_id)
        user = self.dataset.users[user_id]
        product_ids = sorted(self.dataset.products.keys())

        # 각 알고리즘에서 할당된 비율만큼 추천 받기
        n_mf = int(n_recommendations * distribution['mf'])
        n_item_cf = int(n_recommendations * distribution['item_cf'])
        n_content = n_recommendations - n_mf - n_item_cf  # 나머지

        recommendations = []

        # MF에서 추천
        mf_recs = self.mf.recommend(user_idx, n_mf)
        for idx, score in mf_recs:
            recommendations.append((product_ids[idx], score))

        # Item-based CF에서 추천
        item_cf_recs = self.item_cf.recommend(user_idx, n_item_cf + 10)
        # 중복 제거
        existing_ids = {prod_id for prod_id, _ in recommendations}
        for idx, score in item_cf_recs:
            prod_id = product_ids[idx]
            if prod_id not in existing_ids:
                recommendations.append((prod_id, score))
                existing_ids.add(prod_id)
                if len([r for r in recommendations if r[0] not in [p[0] for p in recommendations[:len(recommendations)-1]]]) >= n_mf + n_item_cf:
                    break

        # Content-based에서 추천
        content_recs = self.content_filter.recommend(
            user,
            self.dataset.rating_matrix,
            user_idx,
            n_content + 10
        )
        for prod_id, score in content_recs:
            if prod_id not in existing_ids:
                recommendations.append((prod_id, score))
                existing_ids.add(prod_id)
                if len(recommendations) >= n_recommendations:
                    break

        return recommendations[:n_recommendations]
