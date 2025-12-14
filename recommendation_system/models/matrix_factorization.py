"""
행렬 분해(Matrix Factorization) 기반 추천 모델
- SVD (Singular Value Decomposition)
- ALS (Alternating Least Squares)
- Neural Network based Matrix Factorization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import TruncatedSVD
import warnings


@dataclass
class MFRecommendation:
    """행렬 분해 추천 결과"""
    product_name: str
    predicted_score: float
    latent_factors: np.ndarray


class MatrixFactorization:
    """
    행렬 분해 기반 추천 모델

    주요 기능:
    - SVD 기반 잠재 요인 추출
    - ALS 기반 행렬 분해
    - 사용자/상품 임베딩 학습
    """

    def __init__(
        self,
        n_factors: int = 20,
        learning_rate: float = 0.01,
        regularization: float = 0.02,
        n_epochs: int = 100,
        method: str = 'als'
    ):
        """
        Args:
            n_factors: 잠재 요인 수
            learning_rate: 학습률 (SGD용)
            regularization: 정규화 계수
            n_epochs: 학습 에포크 수
            method: 학습 방법 ('svd', 'als', 'sgd')
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.method = method

        # 학습된 파라미터
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None

        # 매핑
        self.user_ids = None
        self.product_names = None
        self.user_to_idx = None
        self.product_to_idx = None
        self.idx_to_user = None
        self.idx_to_product = None

        self.is_fitted = False

    def fit(self, user_item_matrix: pd.DataFrame, verbose: bool = True):
        """
        모델 학습

        Args:
            user_item_matrix: 사용자-상품 상호작용 매트릭스
            verbose: 학습 과정 출력 여부
        """
        self.user_ids = list(user_item_matrix.index)
        self.product_names = list(user_item_matrix.columns)

        # 인덱스 매핑
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.product_to_idx = {pname: idx for idx, pname in enumerate(self.product_names)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.idx_to_product = {idx: pname for pname, idx in self.product_to_idx.items()}

        matrix = user_item_matrix.values
        n_users, n_items = matrix.shape

        # 전역 평균 (0이 아닌 값만)
        non_zero_mask = matrix > 0
        self.global_bias = matrix[non_zero_mask].mean() if non_zero_mask.sum() > 0 else 0

        if self.method == 'svd':
            self._fit_svd(matrix, verbose)
        elif self.method == 'als':
            self._fit_als(matrix, verbose)
        else:  # sgd
            self._fit_sgd(matrix, verbose)

        self.is_fitted = True
        print(f"Matrix Factorization fitted: {n_users} users, {n_items} items, {self.n_factors} factors")

    def _fit_svd(self, matrix: np.ndarray, verbose: bool):
        """SVD 기반 학습"""
        # 결측치를 전역 평균으로 대체
        filled_matrix = matrix.copy()
        filled_matrix[filled_matrix == 0] = self.global_bias

        # Truncated SVD 적용
        svd = TruncatedSVD(n_components=min(self.n_factors, min(matrix.shape) - 1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.user_factors = svd.fit_transform(filled_matrix)
            self.item_factors = svd.components_.T

        # 바이어스 초기화 (SVD는 바이어스 학습 안 함)
        n_users, n_items = matrix.shape
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        if verbose:
            explained_var = svd.explained_variance_ratio_.sum()
            print(f"SVD explained variance: {explained_var:.4f}")

    def _fit_als(self, matrix: np.ndarray, verbose: bool):
        """ALS 기반 학습"""
        n_users, n_items = matrix.shape

        # 초기화
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # 관측된 항목 마스크
        observed = matrix > 0

        for epoch in range(self.n_epochs):
            # 사용자 요인 업데이트
            for u in range(n_users):
                observed_items = np.where(observed[u])[0]
                if len(observed_items) == 0:
                    continue

                # 관측된 아이템의 요인 행렬
                V_observed = self.item_factors[observed_items]
                ratings = matrix[u, observed_items] - self.global_bias - self.item_bias[observed_items]

                # ALS 업데이트 (정규화 포함)
                A = V_observed.T @ V_observed + self.regularization * np.eye(self.n_factors)
                b = V_observed.T @ ratings

                self.user_factors[u] = np.linalg.solve(A, b)
                self.user_bias[u] = np.mean(ratings - V_observed @ self.user_factors[u])

            # 아이템 요인 업데이트
            for i in range(n_items):
                observed_users = np.where(observed[:, i])[0]
                if len(observed_users) == 0:
                    continue

                U_observed = self.user_factors[observed_users]
                ratings = matrix[observed_users, i] - self.global_bias - self.user_bias[observed_users]

                A = U_observed.T @ U_observed + self.regularization * np.eye(self.n_factors)
                b = U_observed.T @ ratings

                self.item_factors[i] = np.linalg.solve(A, b)
                self.item_bias[i] = np.mean(ratings - U_observed @ self.item_factors[i])

            if verbose and (epoch + 1) % 10 == 0:
                loss = self._compute_loss(matrix, observed)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")

    def _fit_sgd(self, matrix: np.ndarray, verbose: bool):
        """SGD 기반 학습"""
        n_users, n_items = matrix.shape

        # 초기화
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # 관측된 (user, item) 쌍
        observed_pairs = list(zip(*np.where(matrix > 0)))

        for epoch in range(self.n_epochs):
            np.random.shuffle(observed_pairs)

            for u, i in observed_pairs:
                # 예측
                pred = (
                    self.global_bias +
                    self.user_bias[u] +
                    self.item_bias[i] +
                    np.dot(self.user_factors[u], self.item_factors[i])
                )

                # 오차
                error = matrix[u, i] - pred

                # 그래디언트 업데이트
                self.user_bias[u] += self.learning_rate * (error - self.regularization * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.regularization * self.item_bias[i])

                user_factors_old = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (
                    error * self.item_factors[i] - self.regularization * self.user_factors[u]
                )
                self.item_factors[i] += self.learning_rate * (
                    error * user_factors_old - self.regularization * self.item_factors[i]
                )

            if verbose and (epoch + 1) % 10 == 0:
                loss = self._compute_loss(matrix, matrix > 0)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")

    def _compute_loss(self, matrix: np.ndarray, observed: np.ndarray) -> float:
        """손실 함수 계산"""
        predictions = (
            self.global_bias +
            self.user_bias[:, np.newaxis] +
            self.item_bias[np.newaxis, :] +
            self.user_factors @ self.item_factors.T
        )

        errors = (matrix - predictions) * observed
        mse = np.sum(errors ** 2) / observed.sum()

        # 정규화 항
        reg_term = self.regularization * (
            np.sum(self.user_factors ** 2) +
            np.sum(self.item_factors ** 2) +
            np.sum(self.user_bias ** 2) +
            np.sum(self.item_bias ** 2)
        )

        return mse + reg_term

    def predict(self, user_id: int, product_name: str) -> float:
        """
        평점 예측

        Args:
            user_id: 사용자 ID
            product_name: 상품명

        Returns:
            예측 평점
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.user_to_idx:
            return self.global_bias

        if product_name not in self.product_to_idx:
            return self.global_bias

        u = self.user_to_idx[user_id]
        i = self.product_to_idx[product_name]

        pred = (
            self.global_bias +
            self.user_bias[u] +
            self.item_bias[i] +
            np.dot(self.user_factors[u], self.item_factors[i])
        )

        # 점수 범위 제한
        return max(1, min(5, pred))

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_purchased: bool = True,
        user_item_matrix: pd.DataFrame = None
    ) -> List[MFRecommendation]:
        """
        사용자에게 상품 추천

        Args:
            user_id: 사용자 ID
            top_n: 추천 상품 수
            exclude_purchased: 구매한 상품 제외 여부
            user_item_matrix: 원본 매트릭스 (구매 내역 확인용)

        Returns:
            추천 결과 리스트
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.user_to_idx:
            # 콜드 스타트: 인기 상품 추천
            return self._recommend_popular(top_n)

        u = self.user_to_idx[user_id]

        # 모든 상품에 대한 예측 점수
        predictions = (
            self.global_bias +
            self.user_bias[u] +
            self.item_bias +
            self.user_factors[u] @ self.item_factors.T
        )

        # 구매한 상품 제외
        if exclude_purchased and user_item_matrix is not None:
            for i, product_name in enumerate(self.product_names):
                if user_item_matrix.loc[user_id, product_name] > 0:
                    predictions[i] = -np.inf

        # 상위 N개 선택
        top_indices = np.argsort(predictions)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            if predictions[idx] == -np.inf:
                continue

            recommendations.append(MFRecommendation(
                product_name=self.product_names[idx],
                predicted_score=max(1, min(5, predictions[idx])),
                latent_factors=self.item_factors[idx]
            ))

        return recommendations

    def _recommend_popular(self, top_n: int) -> List[MFRecommendation]:
        """인기 상품 추천 (콜드 스타트)"""
        # 아이템 바이어스가 높은 상품 추천
        top_indices = np.argsort(self.item_bias)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            recommendations.append(MFRecommendation(
                product_name=self.product_names[idx],
                predicted_score=self.global_bias + self.item_bias[idx],
                latent_factors=self.item_factors[idx]
            ))

        return recommendations

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """사용자 임베딩 벡터 반환"""
        if user_id not in self.user_to_idx:
            return None

        u = self.user_to_idx[user_id]
        return self.user_factors[u]

    def get_item_embedding(self, product_name: str) -> Optional[np.ndarray]:
        """상품 임베딩 벡터 반환"""
        if product_name not in self.product_to_idx:
            return None

        i = self.product_to_idx[product_name]
        return self.item_factors[i]

    def get_similar_products(
        self,
        product_name: str,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        잠재 요인 기반 유사 상품 검색

        Args:
            product_name: 상품명
            top_n: 반환할 유사 상품 수

        Returns:
            (상품명, 유사도) 튜플 리스트
        """
        item_embedding = self.get_item_embedding(product_name)
        if item_embedding is None:
            return []

        # 코사인 유사도 계산
        similarities = []
        for i, other_name in enumerate(self.product_names):
            if other_name == product_name:
                continue

            other_embedding = self.item_factors[i]
            dot_product = np.dot(item_embedding, other_embedding)
            norm_product = np.linalg.norm(item_embedding) * np.linalg.norm(other_embedding)

            if norm_product > 0:
                similarity = dot_product / norm_product
            else:
                similarity = 0

            similarities.append((other_name, similarity))

        # 유사도 기준 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]

    def get_similar_users(
        self,
        user_id: int,
        top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        잠재 요인 기반 유사 사용자 검색

        Args:
            user_id: 사용자 ID
            top_n: 반환할 유사 사용자 수

        Returns:
            (사용자 ID, 유사도) 튜플 리스트
        """
        user_embedding = self.get_user_embedding(user_id)
        if user_embedding is None:
            return []

        similarities = []
        for u, other_id in enumerate(self.user_ids):
            if other_id == user_id:
                continue

            other_embedding = self.user_factors[u]
            dot_product = np.dot(user_embedding, other_embedding)
            norm_product = np.linalg.norm(user_embedding) * np.linalg.norm(other_embedding)

            if norm_product > 0:
                similarity = dot_product / norm_product
            else:
                similarity = 0

            similarities.append((other_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]

    def save_model(self, filepath: str):
        """모델 저장"""
        np.savez(
            filepath,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_bias=self.user_bias,
            item_bias=self.item_bias,
            global_bias=self.global_bias,
            user_ids=np.array(self.user_ids),
            product_names=np.array(self.product_names)
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """모델 로드"""
        data = np.load(filepath, allow_pickle=True)

        self.user_factors = data['user_factors']
        self.item_factors = data['item_factors']
        self.user_bias = data['user_bias']
        self.item_bias = data['item_bias']
        self.global_bias = data['global_bias'].item()
        self.user_ids = data['user_ids'].tolist()
        self.product_names = data['product_names'].tolist()

        # 매핑 재구성
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.product_to_idx = {pname: idx for idx, pname in enumerate(self.product_names)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.idx_to_product = {idx: pname for pname, idx in self.product_to_idx.items()}

        self.n_factors = self.user_factors.shape[1]
        self.is_fitted = True

        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from utils.data_processor import DataProcessor, create_sample_dataframe

    # 데이터 준비
    processor = DataProcessor()
    df = create_sample_dataframe()
    processed_df = processor.process_data(df)
    matrix, user_map, product_map = processor.create_user_item_matrix()

    # 행렬 분해 모델 학습
    print("\n=== ALS 방식 학습 ===")
    mf_model = MatrixFactorization(n_factors=5, n_epochs=50, method='als')
    mf_model.fit(matrix, verbose=True)

    # 추천
    for user_id in list(processor.user_profiles.keys())[:2]:
        print(f"\n=== User {user_id} MF 추천 ===")
        recommendations = mf_model.recommend(user_id, top_n=3, user_item_matrix=matrix)
        for rec in recommendations:
            print(f"  {rec.product_name[:30]}... (예측: {rec.predicted_score:.2f})")

    # 유사 상품
    sample_product = list(processor.product_info.keys())[0]
    print(f"\n=== '{sample_product[:30]}...' 잠재 요인 기반 유사 상품 ===")
    similar = mf_model.get_similar_products(sample_product, top_n=3)
    for name, sim in similar:
        print(f"  {name[:30]}... (유사도: {sim:.3f})")
