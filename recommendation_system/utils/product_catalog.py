"""
상품 카탈로그 관리 모듈
- 상품 마스터 데이터 관리
- 상품 상태 (판매중/품절/단종) 관리
- 추천 가능 상품 필터링
"""

import pandas as pd
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field


class ProductStatus(Enum):
    """상품 상태"""
    ACTIVE = "active"           # 판매중
    OUT_OF_STOCK = "out_of_stock"  # 일시 품절
    DISCONTINUED = "discontinued"   # 단종
    COMING_SOON = "coming_soon"     # 출시 예정


@dataclass
class Product:
    """상품 정보"""
    product_id: str
    product_name: str
    status: ProductStatus
    category: str = ""
    brand: str = ""
    price: float = 0.0

    # 상품 속성
    target_skin_types: List[str] = field(default_factory=list)
    target_age_groups: List[str] = field(default_factory=list)
    target_concerns: List[str] = field(default_factory=list)

    # 메타 정보
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 추천 가중치 (프로모션 등에 활용)
    boost_score: float = 1.0

    def is_recommendable(self) -> bool:
        """추천 가능 여부"""
        return self.status == ProductStatus.ACTIVE


class ProductCatalog:
    """
    상품 카탈로그 관리자

    주요 기능:
    - 상품 마스터 데이터 관리
    - 상품 상태 업데이트
    - 추천 가능 상품 필터링
    - 신상품/단종 상품 처리
    """

    def __init__(self):
        self.products: Dict[str, Product] = {}
        self._active_products: Set[str] = set()
        self._discontinued_products: Set[str] = set()

    def load_from_csv(self, filepath: str) -> int:
        """
        CSV 파일에서 상품 카탈로그 로드

        CSV 컬럼:
        - product_id, product_name, status, category, brand, price
        - target_skin_types, target_age_groups, target_concerns
        """
        df = pd.read_csv(filepath)
        count = 0

        for _, row in df.iterrows():
            product = Product(
                product_id=str(row.get('product_id', '')),
                product_name=row.get('product_name', ''),
                status=ProductStatus(row.get('status', 'active')),
                category=row.get('category', ''),
                brand=row.get('brand', ''),
                price=float(row.get('price', 0)),
                target_skin_types=self._parse_list(row.get('target_skin_types', '')),
                target_age_groups=self._parse_list(row.get('target_age_groups', '')),
                target_concerns=self._parse_list(row.get('target_concerns', ''))
            )
            self.add_product(product)
            count += 1

        print(f"Loaded {count} products from catalog")
        return count

    def _parse_list(self, value) -> List[str]:
        """리스트 문자열 파싱"""
        if pd.isna(value) or not value:
            return []
        if isinstance(value, list):
            return value
        return [x.strip() for x in str(value).split(',') if x.strip()]

    def add_product(self, product: Product):
        """상품 추가"""
        self.products[product.product_name] = product

        if product.status == ProductStatus.ACTIVE:
            self._active_products.add(product.product_name)
            self._discontinued_products.discard(product.product_name)
        elif product.status == ProductStatus.DISCONTINUED:
            self._discontinued_products.add(product.product_name)
            self._active_products.discard(product.product_name)

    def update_status(self, product_name: str, new_status: ProductStatus):
        """
        상품 상태 업데이트

        Args:
            product_name: 상품명
            new_status: 새로운 상태
        """
        if product_name not in self.products:
            print(f"Warning: Product '{product_name}' not found")
            return

        product = self.products[product_name]
        old_status = product.status
        product.status = new_status
        product.updated_at = datetime.now()

        # 인덱스 업데이트
        if new_status == ProductStatus.ACTIVE:
            self._active_products.add(product_name)
            self._discontinued_products.discard(product_name)
        elif new_status == ProductStatus.DISCONTINUED:
            self._discontinued_products.add(product_name)
            self._active_products.discard(product_name)
        else:
            self._active_products.discard(product_name)

        print(f"Product '{product_name}': {old_status.value} -> {new_status.value}")

    def discontinue_product(self, product_name: str):
        """상품 단종 처리"""
        self.update_status(product_name, ProductStatus.DISCONTINUED)

    def set_out_of_stock(self, product_name: str):
        """상품 품절 처리"""
        self.update_status(product_name, ProductStatus.OUT_OF_STOCK)

    def reactivate_product(self, product_name: str):
        """상품 재판매 처리"""
        self.update_status(product_name, ProductStatus.ACTIVE)

    def get_active_products(self) -> List[str]:
        """판매중인 상품 목록"""
        return list(self._active_products)

    def get_discontinued_products(self) -> List[str]:
        """단종 상품 목록"""
        return list(self._discontinued_products)

    def is_recommendable(self, product_name: str) -> bool:
        """추천 가능 여부 확인"""
        if product_name not in self.products:
            return False
        return self.products[product_name].is_recommendable()

    def filter_recommendable(self, product_names: List[str]) -> List[str]:
        """추천 가능한 상품만 필터링"""
        return [p for p in product_names if self.is_recommendable(p)]

    def get_product(self, product_name: str) -> Optional[Product]:
        """상품 정보 조회"""
        return self.products.get(product_name)

    def get_new_products(self, days: int = 7) -> List[str]:
        """최근 N일 내 추가된 신상품"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        new_products = []

        for name, product in self.products.items():
            if product.created_at.timestamp() > cutoff and product.is_recommendable():
                new_products.append(name)

        return new_products

    def get_boosted_products(self, min_boost: float = 1.5) -> List[str]:
        """부스트 점수가 높은 상품 (프로모션 등)"""
        return [
            name for name, product in self.products.items()
            if product.boost_score >= min_boost and product.is_recommendable()
        ]

    def set_boost_score(self, product_name: str, boost: float):
        """상품 부스트 점수 설정 (프로모션용)"""
        if product_name in self.products:
            self.products[product_name].boost_score = boost

    def sync_with_reviews(self, review_products: Set[str]):
        """
        리뷰 데이터와 카탈로그 동기화

        - 카탈로그에 없는 상품 경고
        - 리뷰 없는 상품 표시
        """
        catalog_products = set(self.products.keys())

        # 리뷰는 있지만 카탈로그에 없는 상품
        missing_in_catalog = review_products - catalog_products
        if missing_in_catalog:
            print(f"Warning: {len(missing_in_catalog)} products in reviews but not in catalog")
            for p in list(missing_in_catalog)[:5]:
                print(f"  - {p}")

        # 카탈로그에는 있지만 리뷰가 없는 상품 (신상품일 수 있음)
        no_reviews = catalog_products - review_products
        active_no_reviews = [p for p in no_reviews if self.is_recommendable(p)]
        if active_no_reviews:
            print(f"Info: {len(active_no_reviews)} active products without reviews (cold start)")

    def get_stats(self) -> Dict:
        """카탈로그 통계"""
        status_counts = {}
        for product in self.products.values():
            status = product.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total': len(self.products),
            'active': len(self._active_products),
            'discontinued': len(self._discontinued_products),
            'by_status': status_counts
        }


# ============================================================
# 샘플 카탈로그 데이터 생성
# ============================================================

def create_sample_catalog() -> ProductCatalog:
    """샘플 상품 카탈로그 생성"""
    catalog = ProductCatalog()

    products = [
        Product(
            product_id="P001",
            product_name="[홀리데이] 자음생크림 리치 50ml 단품기획세트",
            status=ProductStatus.ACTIVE,
            category="크림",
            brand="설화수",
            price=120000,
            target_skin_types=["dry"],
            target_age_groups=["40s", "50s", "50+"],
            target_concerns=["wrinkle", "elasticity"]
        ),
        Product(
            product_id="P002",
            product_name="[홀리데이] 자음생크림 라이트 50ml",
            status=ProductStatus.ACTIVE,
            category="크림",
            brand="설화수",
            price=110000,
            target_skin_types=["oily", "combination"],
            target_age_groups=["20s", "30s"],
            target_concerns=["oiliness"]
        ),
        Product(
            product_id="P003",
            product_name="설화수 윤조에센스",
            status=ProductStatus.ACTIVE,
            category="에센스",
            brand="설화수",
            price=150000,
            target_skin_types=["dry", "normal", "combination"],
            target_age_groups=["30s", "40s", "50s"],
            target_concerns=["dryness", "elasticity"]
        ),
        Product(
            product_id="P004",
            product_name="설화수 자정미라클에센스",
            status=ProductStatus.ACTIVE,
            category="에센스",
            brand="설화수",
            price=180000,
            target_skin_types=["combination"],
            target_age_groups=["40s", "50s"],
            target_concerns=["whitening"]
        ),
        Product(
            product_id="P005",
            product_name="라네즈 워터뱅크 블루 히알루로닉 크림",
            status=ProductStatus.ACTIVE,
            category="크림",
            brand="라네즈",
            price=45000,
            target_skin_types=["dry", "combination"],
            target_age_groups=["20s", "30s"],
            target_concerns=["dryness"]
        ),
        Product(
            product_id="P006",
            product_name="이니스프리 그린티 시드 세럼",
            status=ProductStatus.ACTIVE,
            category="세럼",
            brand="이니스프리",
            price=32000,
            target_skin_types=["oily", "combination"],
            target_age_groups=["20s", "30s"],
            target_concerns=["trouble", "oiliness"]
        ),
        Product(
            product_id="P007",
            product_name="설화수 자음생크림",
            status=ProductStatus.ACTIVE,
            category="크림",
            brand="설화수",
            price=130000,
            target_skin_types=["dry"],
            target_age_groups=["40s", "50s", "50+"],
            target_concerns=["wrinkle", "elasticity"]
        ),
        Product(
            product_id="P008",
            product_name="이니스프리 제주 화산송이 모공 마스크",
            status=ProductStatus.ACTIVE,
            category="마스크",
            brand="이니스프리",
            price=15000,
            target_skin_types=["oily"],
            target_age_groups=["20s", "30s"],
            target_concerns=["pore", "oiliness"]
        ),
        Product(
            product_id="P009",
            product_name="아이오페 더마 리페어 시카 크림",
            status=ProductStatus.ACTIVE,
            category="크림",
            brand="아이오페",
            price=48000,
            target_skin_types=["sensitive", "oily"],
            target_age_groups=["20s", "30s"],
            target_concerns=["trouble", "sensitivity"]
        ),
        Product(
            product_id="P010",
            product_name="설화수 설린 세럼",
            status=ProductStatus.ACTIVE,
            category="세럼",
            brand="설화수",
            price=200000,
            target_skin_types=["combination", "dry"],
            target_age_groups=["40s", "50s"],
            target_concerns=["whitening"]
        ),
        Product(
            product_id="P011",
            product_name="설화수 타임트레저 리뉴잉 크림",
            status=ProductStatus.ACTIVE,
            category="크림",
            brand="설화수",
            price=350000,
            target_skin_types=["dry"],
            target_age_groups=["40s", "50s", "50+"],
            target_concerns=["wrinkle", "elasticity"]
        ),
        Product(
            product_id="P012",
            product_name="라네즈 워터뱅크 블루 히알루로닉 세럼",
            status=ProductStatus.ACTIVE,
            category="세럼",
            brand="라네즈",
            price=42000,
            target_skin_types=["oily", "combination"],
            target_age_groups=["20s", "30s"],
            target_concerns=["dryness"]
        ),
        # 단종 상품 예시
        Product(
            product_id="P100",
            product_name="[단종] 설화수 진설크림",
            status=ProductStatus.DISCONTINUED,
            category="크림",
            brand="설화수",
            price=0,
        ),
        # 품절 상품 예시
        Product(
            product_id="P101",
            product_name="[한정] 설화수 홀리데이 세트",
            status=ProductStatus.OUT_OF_STOCK,
            category="세트",
            brand="설화수",
            price=250000,
        ),
    ]

    for product in products:
        catalog.add_product(product)

    return catalog


if __name__ == "__main__":
    # 테스트
    catalog = create_sample_catalog()

    print("\n=== 카탈로그 통계 ===")
    stats = catalog.get_stats()
    print(f"총 상품: {stats['total']}")
    print(f"판매중: {stats['active']}")
    print(f"단종: {stats['discontinued']}")

    print("\n=== 판매중 상품 ===")
    for p in catalog.get_active_products()[:5]:
        print(f"  - {p}")

    print("\n=== 단종 처리 테스트 ===")
    catalog.discontinue_product("설화수 윤조에센스")
    print(f"추천 가능: {catalog.is_recommendable('설화수 윤조에센스')}")

    print("\n=== 재판매 처리 ===")
    catalog.reactivate_product("설화수 윤조에센스")
    print(f"추천 가능: {catalog.is_recommendable('설화수 윤조에센스')}")
