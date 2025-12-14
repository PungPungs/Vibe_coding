"""
데이터 모델 정의
- 사용자, 상품, 상호작용 데이터를 관리하는 클래스 정의
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np


# ============================================================================
# 사용자 데이터 모델
# ============================================================================

@dataclass
class User:
    """
    사용자 정보를 담는 데이터 클래스

    Attributes:
        user_id (int): 사용자 고유 ID
        age (int): 나이
        gender (str): 성별 ('M', 'F', 'Other')
        location (str): 거주 지역
        preferences (List[str]): 선호 카테고리 리스트
        purchase_history (List[int]): 구매한 상품 ID 리스트
    """
    user_id: int
    age: int
    gender: str
    location: str
    preferences: List[str]
    purchase_history: List[int]


# ============================================================================
# 상품 데이터 모델
# ============================================================================

@dataclass
class Product:
    """
    상품 정보를 담는 데이터 클래스

    Attributes:
        product_id (int): 상품 고유 ID
        name (str): 상품명
        category (str): 상품 카테고리
        price (float): 가격
        description (str): 상품 설명
        tags (List[str]): 상품 태그 리스트
        features (Dict[str, any]): 상품 특성 (브랜드, 색상 등)
    """
    product_id: int
    name: str
    category: str
    price: float
    description: str
    tags: List[str]
    features: Dict[str, any]


# ============================================================================
# 상호작용 데이터 모델
# ============================================================================

@dataclass
class Interaction:
    """
    사용자-상품 상호작용 데이터 클래스

    Attributes:
        user_id (int): 사용자 ID
        product_id (int): 상품 ID
        rating (Optional[float]): 평점 (1-5)
        interaction_type (str): 상호작용 타입 ('view', 'purchase', 'rating', 'cart')
        timestamp (datetime): 상호작용 발생 시간
        implicit_score (float): 암시적 피드백 점수 (조회수, 체류시간 등 기반)
    """
    user_id: int
    product_id: int
    rating: Optional[float]
    interaction_type: str
    timestamp: datetime
    implicit_score: float = 1.0


# ============================================================================
# 데이터셋 관리 클래스
# ============================================================================

class Dataset:
    """
    전체 데이터셋을 관리하는 클래스
    - 사용자, 상품, 상호작용 데이터를 통합 관리
    - User-Item 매트릭스 생성
    """

    def __init__(self):
        """데이터셋 초기화"""
        self.users: Dict[int, User] = {}
        self.products: Dict[int, Product] = {}
        self.interactions: List[Interaction] = []
        self.rating_matrix: Optional[np.ndarray] = None

    def add_user(self, user: User) -> None:
        """
        사용자 추가

        Args:
            user (User): 추가할 사용자 객체
        """
        self.users[user.user_id] = user

    def add_product(self, product: Product) -> None:
        """
        상품 추가

        Args:
            product (Product): 추가할 상품 객체
        """
        self.products[product.product_id] = product

    def add_interaction(self, interaction: Interaction) -> None:
        """
        상호작용 데이터 추가

        Args:
            interaction (Interaction): 추가할 상호작용 객체
        """
        self.interactions.append(interaction)

    def build_rating_matrix(self) -> np.ndarray:
        """
        User-Item Rating 매트릭스 생성
        - 행: 사용자 ID
        - 열: 상품 ID
        - 값: 평점 또는 암시적 점수

        Returns:
            np.ndarray: Rating 매트릭스 (num_users x num_products)
        """
        num_users = len(self.users)
        num_products = len(self.products)

        # 사용자 ID를 인덱스로 매핑
        user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(self.users.keys()))}
        product_id_to_idx = {pid: idx for idx, pid in enumerate(sorted(self.products.keys()))}

        # 매트릭스 초기화 (0으로 채워짐, 상호작용 없음을 의미)
        matrix = np.zeros((num_users, num_products))

        # 상호작용 데이터로 매트릭스 채우기
        for interaction in self.interactions:
            user_idx = user_id_to_idx.get(interaction.user_id)
            product_idx = product_id_to_idx.get(interaction.product_id)

            if user_idx is not None and product_idx is not None:
                # 명시적 평점이 있으면 사용, 없으면 암시적 점수 사용
                score = interaction.rating if interaction.rating else interaction.implicit_score
                matrix[user_idx][product_idx] = score

        self.rating_matrix = matrix
        return matrix

    def get_user_vector(self, user_id: int) -> Optional[np.ndarray]:
        """
        특정 사용자의 평점 벡터 반환

        Args:
            user_id (int): 사용자 ID

        Returns:
            Optional[np.ndarray]: 사용자 평점 벡터 (없으면 None)
        """
        if self.rating_matrix is None:
            self.build_rating_matrix()

        user_ids = sorted(self.users.keys())
        if user_id in user_ids:
            user_idx = user_ids.index(user_id)
            return self.rating_matrix[user_idx]
        return None

    def get_product_vector(self, product_id: int) -> Optional[np.ndarray]:
        """
        특정 상품의 평점 벡터 반환

        Args:
            product_id (int): 상품 ID

        Returns:
            Optional[np.ndarray]: 상품 평점 벡터 (없으면 None)
        """
        if self.rating_matrix is None:
            self.build_rating_matrix()

        product_ids = sorted(self.products.keys())
        if product_id in product_ids:
            product_idx = product_ids.index(product_id)
            return self.rating_matrix[:, product_idx]
        return None
