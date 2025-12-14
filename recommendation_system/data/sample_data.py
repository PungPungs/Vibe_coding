"""
샘플 데이터 생성기
- 테스트 및 데모를 위한 샘플 사용자, 상품, 상호작용 데이터 생성
"""

from datetime import datetime, timedelta
import random
from typing import List
from data_models import User, Product, Interaction, Dataset


# ============================================================================
# 샘플 데이터 생성 함수
# ============================================================================

def generate_sample_users(num_users: int = 100) -> List[User]:
    """
    샘플 사용자 데이터 생성

    Args:
        num_users (int): 생성할 사용자 수

    Returns:
        List[User]: 생성된 사용자 리스트
    """
    users = []
    locations = ['서울', '부산', '대구', '인천', '광주', '대전', '울산']
    categories = ['전자제품', '패션', '뷰티', '스포츠', '도서', '식품', '가구', '완구']

    for i in range(num_users):
        user = User(
            user_id=i + 1,
            age=random.randint(18, 65),
            gender=random.choice(['M', 'F']),
            location=random.choice(locations),
            preferences=random.sample(categories, k=random.randint(2, 4)),
            purchase_history=[]
        )
        users.append(user)

    return users


def generate_sample_products(num_products: int = 200) -> List[Product]:
    """
    샘플 상품 데이터 생성

    Args:
        num_products (int): 생성할 상품 수

    Returns:
        List[Product]: 생성된 상품 리스트
    """
    products = []

    # 카테고리별 상품 템플릿
    product_templates = {
        '전자제품': {
            'names': ['노트북', '스마트폰', '태블릿', '이어폰', '스마트워치', '카메라'],
            'tags': ['최신', '고성능', '휴대용', '무선', '프리미엄'],
            'price_range': (100000, 2000000)
        },
        '패션': {
            'names': ['티셔츠', '청바지', '운동화', '가디건', '원피스', '코트'],
            'tags': ['트렌디', '편안한', '스타일리시', '캐주얼', '정장'],
            'price_range': (20000, 300000)
        },
        '뷰티': {
            'names': ['립스틱', '아이섀도우', '스킨케어세트', '향수', '마스크팩'],
            'tags': ['자연주의', '보습', '안티에이징', '유기농', '순한'],
            'price_range': (10000, 150000)
        },
        '스포츠': {
            'names': ['요가매트', '덤벨', '러닝화', '운동복', '자전거', '수영복'],
            'tags': ['내구성', '경량', '편안함', '전문가용', '입문용'],
            'price_range': (15000, 500000)
        },
        '도서': {
            'names': ['소설', '자기계발서', '요리책', '만화', '여행서', '전문서적'],
            'tags': ['베스트셀러', '추천', '신간', '스테디셀러', '화제의책'],
            'price_range': (10000, 50000)
        },
        '식품': {
            'names': ['유기농채소', '과일세트', '견과류', '건강식품', '간식', '음료'],
            'tags': ['신선한', '유기농', '건강한', '무첨가', '프리미엄'],
            'price_range': (5000, 100000)
        }
    }

    categories = list(product_templates.keys())

    for i in range(num_products):
        category = random.choice(categories)
        template = product_templates[category]

        product = Product(
            product_id=i + 1,
            name=f"{random.choice(template['names'])} #{i+1}",
            category=category,
            price=random.randint(*template['price_range']),
            description=f"{category} 카테고리의 우수한 상품입니다.",
            tags=random.sample(template['tags'], k=random.randint(2, 3)),
            features={
                'brand': f"Brand_{random.randint(1, 20)}",
                'color': random.choice(['블랙', '화이트', '네이비', '레드', '그레이']),
                'rating_avg': round(random.uniform(3.5, 5.0), 1)
            }
        )
        products.append(product)

    return products


def generate_sample_interactions(
    users: List[User],
    products: List[Product],
    interactions_per_user: int = 20
) -> List[Interaction]:
    """
    샘플 상호작용 데이터 생성

    Args:
        users (List[User]): 사용자 리스트
        products (List[Product]): 상품 리스트
        interactions_per_user (int): 사용자당 평균 상호작용 수

    Returns:
        List[Interaction]: 생성된 상호작용 리스트
    """
    interactions = []
    interaction_types = ['view', 'purchase', 'rating', 'cart']

    for user in users:
        num_interactions = random.randint(
            interactions_per_user - 5,
            interactions_per_user + 5
        )

        # 사용자 선호도에 맞는 상품 선택 확률 높이기
        preferred_products = [
            p for p in products
            if p.category in user.preferences
        ]

        # 선호 카테고리 70%, 기타 카테고리 30%
        selected_products = []
        for _ in range(num_interactions):
            if random.random() < 0.7 and preferred_products:
                selected_products.append(random.choice(preferred_products))
            else:
                selected_products.append(random.choice(products))

        for product in selected_products:
            interaction_type = random.choice(interaction_types)

            # 상호작용 타입에 따라 평점 및 암시적 점수 설정
            rating = None
            implicit_score = 1.0

            if interaction_type == 'rating':
                rating = random.choice([3.0, 3.5, 4.0, 4.5, 5.0])
                implicit_score = rating
            elif interaction_type == 'purchase':
                rating = random.choice([4.0, 4.5, 5.0])
                implicit_score = 5.0
            elif interaction_type == 'cart':
                implicit_score = 3.0
            elif interaction_type == 'view':
                implicit_score = 1.0

            # 타임스탬프: 최근 30일 이내
            days_ago = random.randint(0, 30)
            timestamp = datetime.now() - timedelta(days=days_ago)

            interaction = Interaction(
                user_id=user.user_id,
                product_id=product.product_id,
                rating=rating,
                interaction_type=interaction_type,
                timestamp=timestamp,
                implicit_score=implicit_score
            )
            interactions.append(interaction)

            # 구매한 경우 사용자 구매 이력에 추가
            if interaction_type == 'purchase':
                user.purchase_history.append(product.product_id)

    return interactions


def create_sample_dataset() -> Dataset:
    """
    완전한 샘플 데이터셋 생성

    Returns:
        Dataset: 초기화된 데이터셋 객체
    """
    dataset = Dataset()

    # 사용자 생성 및 추가
    users = generate_sample_users(num_users=100)
    for user in users:
        dataset.add_user(user)

    # 상품 생성 및 추가
    products = generate_sample_products(num_products=200)
    for product in products:
        dataset.add_product(product)

    # 상호작용 생성 및 추가
    interactions = generate_sample_interactions(users, products, interactions_per_user=20)
    for interaction in interactions:
        dataset.add_interaction(interaction)

    # Rating 매트릭스 빌드
    dataset.build_rating_matrix()

    return dataset


# ============================================================================
# 메인 실행부 (테스트용)
# ============================================================================

if __name__ == "__main__":
    print("샘플 데이터셋 생성 중...")
    dataset = create_sample_dataset()

    print(f"\n생성 완료!")
    print(f"- 사용자 수: {len(dataset.users)}")
    print(f"- 상품 수: {len(dataset.products)}")
    print(f"- 상호작용 수: {len(dataset.interactions)}")
    print(f"- Rating 매트릭스 크기: {dataset.rating_matrix.shape}")

    # 샘플 사용자 정보 출력
    sample_user = list(dataset.users.values())[0]
    print(f"\n샘플 사용자 정보:")
    print(f"  ID: {sample_user.user_id}")
    print(f"  나이: {sample_user.age}, 성별: {sample_user.gender}")
    print(f"  지역: {sample_user.location}")
    print(f"  선호 카테고리: {sample_user.preferences}")
