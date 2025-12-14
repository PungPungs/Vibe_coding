"""
개인화된 하이브리드 추천 시스템 - 메인 실행 파일
사용자 정보를 입력받아 맞춤형 상품을 추천합니다.
"""

import sys
import os
import numpy as np

# 모듈 import를 위한 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_models import Dataset, User
from data.sample_data import create_sample_dataset
from models.hybrid_recommender import (
    WeightedHybridRecommender,
    SwitchingHybridRecommender,
    MixedHybridRecommender
)


# ============================================================================
# 추천 결과 출력 함수
# ============================================================================

def display_recommendations(
    recommendations: list,
    dataset: Dataset,
    title: str = "추천 상품"
):
    """
    추천 결과를 보기 좋게 출력

    Args:
        recommendations (list): 추천 결과 리스트 [(product_id, score), ...]
        dataset (Dataset): 데이터셋
        title (str): 출력 제목
    """
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}\n")

    if not recommendations:
        print("추천할 상품이 없습니다.")
        return

    for i, (product_id, score) in enumerate(recommendations, 1):
        product = dataset.products[product_id]
        print(f"{i}. [{product.category}] {product.name}")
        print(f"   가격: {product.price:,}원 | 추천 점수: {score:.4f}")
        print(f"   태그: {', '.join(product.tags)}")
        print(f"   브랜드: {product.features['brand']} | 평균 평점: {product.features['rating_avg']}")
        print()


def display_user_info(user: User, dataset: Dataset):
    """
    사용자 정보 출력

    Args:
        user (User): 사용자 객체
        dataset (Dataset): 데이터셋
    """
    print(f"\n{'=' * 80}")
    print(f"사용자 정보 (ID: {user.user_id})")
    print(f"{'=' * 80}")
    print(f"나이: {user.age}세 | 성별: {user.gender} | 지역: {user.location}")
    print(f"선호 카테고리: {', '.join(user.preferences)}")

    # 평가 이력
    user_ids = sorted(dataset.users.keys())
    user_idx = user_ids.index(user.user_id)
    user_ratings = dataset.rating_matrix[user_idx]
    num_ratings = np.count_nonzero(user_ratings)
    avg_rating = user_ratings[user_ratings > 0].mean() if num_ratings > 0 else 0

    print(f"평가한 상품 수: {num_ratings}개 | 평균 평점: {avg_rating:.2f}")

    # 최근 구매 상품
    if user.purchase_history:
        print(f"\n최근 구매 상품:")
        for i, prod_id in enumerate(user.purchase_history[:5], 1):
            product = dataset.products[prod_id]
            print(f"  {i}. {product.name} ({product.category})")


# ============================================================================
# 메인 추천 시스템 실행
# ============================================================================

def run_recommendation_system():
    """
    추천 시스템 메인 실행 함수
    """
    print("\n" + "=" * 80)
    print("개인화된 하이브리드 추천 시스템".center(80))
    print("=" * 80)

    # ========================================
    # 1. 데이터셋 로드
    # ========================================
    print("\n[1단계] 데이터셋 로딩 중...")
    dataset = create_sample_dataset()
    print(f"✓ 사용자: {len(dataset.users)}명")
    print(f"✓ 상품: {len(dataset.products)}개")
    print(f"✓ 상호작용: {len(dataset.interactions)}건")

    # ========================================
    # 2. 하이브리드 추천기 초기화
    # ========================================
    print("\n[2단계] 하이브리드 추천 엔진 초기화 중...")
    print("  - User-based Collaborative Filtering")
    print("  - Item-based Collaborative Filtering")
    print("  - Matrix Factorization (SVD)")
    print("  - Content-based Filtering (TF-IDF + Feature-based)")

    weighted_recommender = WeightedHybridRecommender(dataset)
    switching_recommender = SwitchingHybridRecommender(dataset)
    mixed_recommender = MixedHybridRecommender(dataset)

    print("✓ 모든 추천 엔진 초기화 완료")

    # ========================================
    # 3. 테스트 사용자 선택
    # ========================================
    print("\n[3단계] 테스트 사용자 선택")

    # 다양한 사용자 프로필 선택
    user_ids = sorted(dataset.users.keys())

    # Cold start 사용자 (평가 적음)
    cold_start_user_id = None
    for uid in user_ids:
        user_idx = user_ids.index(uid)
        if np.count_nonzero(dataset.rating_matrix[user_idx]) < 5:
            cold_start_user_id = uid
            break

    # 활발한 사용자 (평가 많음)
    active_user_id = None
    for uid in user_ids:
        user_idx = user_ids.index(uid)
        if np.count_nonzero(dataset.rating_matrix[user_idx]) >= 20:
            active_user_id = uid
            break

    # 중간 사용자
    normal_user_id = user_ids[10] if len(user_ids) > 10 else user_ids[0]

    # ========================================
    # 4. 추천 실행 및 결과 출력
    # ========================================

    test_cases = []
    if cold_start_user_id:
        test_cases.append(("Cold Start 사용자", cold_start_user_id))
    if active_user_id:
        test_cases.append(("활발한 사용자", active_user_id))
    test_cases.append(("일반 사용자", normal_user_id))

    for user_type, user_id in test_cases:
        user = dataset.users[user_id]

        print("\n" + "=" * 80)
        print(f"테스트 케이스: {user_type}")
        print("=" * 80)

        display_user_info(user, dataset)

        # 가중 평균 하이브리드
        print(f"\n{'─' * 80}")
        print("[방법 1] 가중 평균 하이브리드 추천")
        print("  → 모든 알고리즘의 결과를 가중치로 결합")
        print(f"{'─' * 80}")
        weighted_recs = weighted_recommender.recommend(user_id, n_recommendations=5)
        display_recommendations(weighted_recs, dataset, "가중 평균 하이브리드 추천 결과")

        # 스위칭 하이브리드
        print(f"\n{'─' * 80}")
        print("[방법 2] 스위칭 하이브리드 추천")
        print("  → 사용자 상태에 따라 최적의 알고리즘 선택")
        print(f"{'─' * 80}")
        switching_recs = switching_recommender.recommend(user_id, n_recommendations=5)
        display_recommendations(switching_recs, dataset, "스위칭 하이브리드 추천 결과")

        # 혼합 하이브리드
        print(f"\n{'─' * 80}")
        print("[방법 3] 혼합 하이브리드 추천")
        print("  → 여러 알고리즘에서 일부씩 추천하여 다양성 확보")
        print(f"{'─' * 80}")
        mixed_recs = mixed_recommender.recommend(user_id, n_recommendations=5)
        display_recommendations(mixed_recs, dataset, "혼합 하이브리드 추천 결과")

    # ========================================
    # 5. 인터랙티브 모드
    # ========================================
    print("\n" + "=" * 80)
    print("인터랙티브 모드")
    print("=" * 80)

    while True:
        print("\n사용자 ID를 입력하세요 (1-100, 종료: 0):")
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            input_id = int(user_input)

            if input_id == 0:
                print("\n추천 시스템을 종료합니다.")
                break

            if input_id not in dataset.users:
                print(f"오류: 사용자 ID {input_id}를 찾을 수 없습니다. (범위: 1-{len(dataset.users)})")
                continue

            user = dataset.users[input_id]
            display_user_info(user, dataset)

            print("\n추천 방법을 선택하세요:")
            print("  1. 가중 평균 하이브리드")
            print("  2. 스위칭 하이브리드")
            print("  3. 혼합 하이브리드")

            method = input("> ").strip()

            if method == "1":
                recs = weighted_recommender.recommend(input_id, n_recommendations=10)
                display_recommendations(recs, dataset, "가중 평균 하이브리드 추천 결과")
            elif method == "2":
                recs = switching_recommender.recommend(input_id, n_recommendations=10)
                display_recommendations(recs, dataset, "스위칭 하이브리드 추천 결과")
            elif method == "3":
                recs = mixed_recommender.recommend(input_id, n_recommendations=10)
                display_recommendations(recs, dataset, "혼합 하이브리드 추천 결과")
            else:
                print("잘못된 선택입니다.")

        except ValueError:
            print("오류: 올바른 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n\n추천 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")


# ============================================================================
# 프로그램 엔트리 포인트
# ============================================================================

if __name__ == "__main__":
    try:
        run_recommendation_system()
    except Exception as e:
        print(f"\n시스템 오류: {e}")
        import traceback
        traceback.print_exc()
