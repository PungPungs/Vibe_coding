"""
협업 필터링 (Collaborative Filtering) 모듈
- User-based CF: 유사한 사용자들의 선호도를 기반으로 추천
- Item-based CF: 유사한 상품들의 패턴을 기반으로 추천
- Matrix Factorization: SVD를 이용한 잠재 요인 분해
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


# ============================================================================
# 유틸리티 함수
# ============================================================================

def calculate_similarity(matrix: np.ndarray, method: str = 'cosine') -> np.ndarray:
    """
    유사도 계산 함수

    Args:
        matrix (np.ndarray): 입력 매트릭스
        method (str): 유사도 계산 방법 ('cosine', 'pearson')

    Returns:
        np.ndarray: 유사도 매트릭스
    """
    if method == 'cosine':
        # 코사인 유사도: 벡터 간 각도 기반
        # 값 범위: -1 ~ 1 (1에 가까울수록 유사)
        return cosine_similarity(matrix)
    elif method == 'pearson':
        # 피어슨 상관계수: 선형 상관관계 기반
        # 값 범위: -1 ~ 1 (1에 가까울수록 양의 상관관계)
        return np.corrcoef(matrix)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


# ============================================================================
# User-based Collaborative Filtering
# ============================================================================

class UserBasedCF:
    """
    사용자 기반 협업 필터링
    - 타겟 사용자와 유사한 사용자들을 찾아 그들의 선호 상품 추천
    - Cold Start 문제: 신규 사용자에 대한 추천 어려움
    - Scalability: 사용자 수가 많을 때 계산량 증가
    """

    def __init__(self, rating_matrix: np.ndarray, similarity_method: str = 'cosine'):
        """
        초기화

        Args:
            rating_matrix (np.ndarray): User-Item 평점 매트릭스 (num_users x num_items)
            similarity_method (str): 유사도 계산 방법
        """
        self.rating_matrix = rating_matrix
        self.similarity_method = similarity_method

        # 사용자 간 유사도 매트릭스 계산
        # shape: (num_users x num_users)
        self.user_similarity = calculate_similarity(rating_matrix, similarity_method)

    def recommend(
        self,
        user_idx: int,
        n_recommendations: int = 10,
        k_neighbors: int = 10
    ) -> List[Tuple[int, float]]:
        """
        특정 사용자에게 상품 추천

        추천 프로세스:
        1. 타겟 사용자와 유사한 k명의 이웃 찾기
        2. 이웃들의 평점을 유사도로 가중 평균
        3. 타겟 사용자가 아직 평가하지 않은 상품 중 상위 n개 추천

        Args:
            user_idx (int): 타겟 사용자 인덱스
            n_recommendations (int): 추천할 상품 개수
            k_neighbors (int): 고려할 이웃 사용자 수

        Returns:
            List[Tuple[int, float]]: (상품 인덱스, 예측 평점) 리스트
        """
        # 1. 타겟 사용자와 다른 모든 사용자 간의 유사도 가져오기
        user_similarities = self.user_similarity[user_idx].copy()

        # 자기 자신은 제외
        user_similarities[user_idx] = -1

        # 2. 상위 k명의 유사한 이웃 찾기
        neighbor_indices = np.argsort(user_similarities)[::-1][:k_neighbors]
        neighbor_similarities = user_similarities[neighbor_indices]

        # 3. 이웃들의 평점을 유사도로 가중 평균하여 예측 평점 계산
        # weighted_sum: 유사도 * 평점의 합
        # similarity_sum: 유사도의 합 (정규화용)
        weighted_sum = np.zeros(self.rating_matrix.shape[1])
        similarity_sum = np.zeros(self.rating_matrix.shape[1])

        for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
            if similarity > 0:  # 양의 유사도만 고려
                neighbor_ratings = self.rating_matrix[neighbor_idx]
                # 평점이 있는 상품에 대해서만 계산
                mask = neighbor_ratings > 0
                weighted_sum[mask] += similarity * neighbor_ratings[mask]
                similarity_sum[mask] += similarity

        # 예측 평점 = 가중 평균
        predicted_ratings = np.zeros(self.rating_matrix.shape[1])
        nonzero_mask = similarity_sum > 0
        predicted_ratings[nonzero_mask] = weighted_sum[nonzero_mask] / similarity_sum[nonzero_mask]

        # 4. 이미 평가한 상품 제외
        user_ratings = self.rating_matrix[user_idx]
        predicted_ratings[user_ratings > 0] = 0

        # 5. 상위 n개 상품 선택
        top_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        recommendations = [
            (idx, predicted_ratings[idx])
            for idx in top_indices
            if predicted_ratings[idx] > 0
        ]

        return recommendations


# ============================================================================
# Item-based Collaborative Filtering
# ============================================================================

class ItemBasedCF:
    """
    아이템 기반 협업 필터링
    - 타겟 상품과 유사한 상품들을 찾아 사용자가 좋아할 상품 추천
    - User-based보다 안정적 (상품 특성은 잘 변하지 않음)
    - 계산 효율성: 상품 수가 사용자 수보다 적을 때 유리
    """

    def __init__(self, rating_matrix: np.ndarray, similarity_method: str = 'cosine'):
        """
        초기화

        Args:
            rating_matrix (np.ndarray): User-Item 평점 매트릭스 (num_users x num_items)
            similarity_method (str): 유사도 계산 방법
        """
        self.rating_matrix = rating_matrix
        self.similarity_method = similarity_method

        # 아이템 간 유사도 매트릭스 계산
        # Transpose하여 (num_items x num_users)로 변환 후 유사도 계산
        # shape: (num_items x num_items)
        self.item_similarity = calculate_similarity(rating_matrix.T, similarity_method)

    def recommend(
        self,
        user_idx: int,
        n_recommendations: int = 10,
        k_neighbors: int = 10
    ) -> List[Tuple[int, float]]:
        """
        특정 사용자에게 상품 추천

        추천 프로세스:
        1. 사용자가 높게 평가한 상품들 찾기
        2. 각 상품과 유사한 k개의 상품 찾기
        3. 유사도로 가중 평균하여 예측 평점 계산
        4. 상위 n개 상품 추천

        Args:
            user_idx (int): 타겟 사용자 인덱스
            n_recommendations (int): 추천할 상품 개수
            k_neighbors (int): 고려할 유사 상품 수

        Returns:
            List[Tuple[int, float]]: (상품 인덱스, 예측 평점) 리스트
        """
        user_ratings = self.rating_matrix[user_idx]

        # 예측 평점 초기화
        predicted_ratings = np.zeros(len(user_ratings))

        # 각 상품에 대해 예측
        for item_idx in range(len(user_ratings)):
            # 이미 평가한 상품은 스킵
            if user_ratings[item_idx] > 0:
                continue

            # 타겟 상품과 유사한 상품들 찾기
            item_similarities = self.item_similarity[item_idx].copy()
            item_similarities[item_idx] = -1  # 자기 자신 제외

            # 사용자가 평가한 상품 중에서만 유사 상품 선택
            rated_mask = user_ratings > 0
            item_similarities[~rated_mask] = -1

            # 상위 k개 유사 상품 선택
            similar_items = np.argsort(item_similarities)[::-1][:k_neighbors]
            similar_scores = item_similarities[similar_items]

            # 가중 평균으로 예측 평점 계산
            if similar_scores[similar_scores > 0].sum() > 0:
                weighted_sum = (similar_scores * user_ratings[similar_items]).sum()
                similarity_sum = similar_scores.sum()
                predicted_ratings[item_idx] = weighted_sum / similarity_sum

        # 상위 n개 상품 선택
        top_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        recommendations = [
            (idx, predicted_ratings[idx])
            for idx in top_indices
            if predicted_ratings[idx] > 0
        ]

        return recommendations


# ============================================================================
# Matrix Factorization (SVD)
# ============================================================================

class MatrixFactorization:
    """
    행렬 분해 기반 협업 필터링
    - SVD (Singular Value Decomposition)를 사용한 차원 축소
    - 잠재 요인 (Latent Factors) 추출로 노이즈 감소
    - Sparsity 문제 완화
    """

    def __init__(self, rating_matrix: np.ndarray, n_factors: int = 20):
        """
        초기화 및 SVD 수행

        SVD: R ≈ U * Σ * V^T
        - U: User latent factors (num_users x n_factors)
        - Σ: Singular values (n_factors x n_factors)
        - V^T: Item latent factors (n_factors x num_items)

        Args:
            rating_matrix (np.ndarray): User-Item 평점 매트릭스
            n_factors (int): 잠재 요인 개수 (차원)
        """
        self.rating_matrix = rating_matrix
        self.n_factors = n_factors

        # 평점 평균 계산 (정규화용)
        # 0인 값(평가 안 한 상품)은 제외
        nonzero_mask = rating_matrix > 0
        self.mean_rating = rating_matrix[nonzero_mask].mean() if nonzero_mask.any() else 0

        # 평점 정규화 (평균 제거)
        normalized_matrix = rating_matrix.copy()
        normalized_matrix[nonzero_mask] -= self.mean_rating

        # SVD 수행
        # svds: sparse SVD (큰 행렬에 효율적)
        self.U, self.sigma, self.Vt = svds(normalized_matrix, k=n_factors)

        # 예측 매트릭스 계산
        # shape: (num_users x num_items)
        self.predicted_matrix = np.dot(
            np.dot(self.U, np.diag(self.sigma)),
            self.Vt
        ) + self.mean_rating

    def recommend(
        self,
        user_idx: int,
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        특정 사용자에게 상품 추천

        Args:
            user_idx (int): 타겟 사용자 인덱스
            n_recommendations (int): 추천할 상품 개수

        Returns:
            List[Tuple[int, float]]: (상품 인덱스, 예측 평점) 리스트
        """
        # 예측된 평점 가져오기
        predicted_ratings = self.predicted_matrix[user_idx].copy()

        # 이미 평가한 상품 제외
        user_ratings = self.rating_matrix[user_idx]
        predicted_ratings[user_ratings > 0] = 0

        # 상위 n개 상품 선택
        top_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        recommendations = [
            (idx, predicted_ratings[idx])
            for idx in top_indices
            if predicted_ratings[idx] > 0
        ]

        return recommendations

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """
        사용자의 잠재 요인 벡터 반환

        Args:
            user_idx (int): 사용자 인덱스

        Returns:
            np.ndarray: 사용자 임베딩 벡터 (n_factors,)
        """
        return self.U[user_idx] * self.sigma

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """
        상품의 잠재 요인 벡터 반환

        Args:
            item_idx (int): 상품 인덱스

        Returns:
            np.ndarray: 상품 임베딩 벡터 (n_factors,)
        """
        return self.Vt[:, item_idx]
