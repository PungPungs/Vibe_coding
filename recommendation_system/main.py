"""
개인화된 하이브리드 상품 추천 시스템
======================================

사용자 정보(피부 타입, 연령대, 피부 고민 등)를 기반으로
협업 필터링, 콘텐츠 기반 필터링, 행렬 분해를 결합한
하이브리드 추천 시스템

주요 기능:
- 개인화된 상품 추천
- 유사 상품 추천
- 인기/트렌딩 상품 추천
- 추천 이유 설명

사용법:
    python main.py
"""

import os
import sys

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.data_processor import DataProcessor, create_sample_dataframe
from utils.feature_extractor import FeatureExtractor
from models.hybrid_recommender import HybridRecommender


def print_header(title: str, width: int = 60):
    """섹션 헤더 출력"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_recommendation(rec, idx: int):
    """추천 결과 출력"""
    print(f"\n  {idx}. {rec.product_name}")
    print(f"     종합 점수: {rec.final_score:.2f}/5.00")
    print(f"     개별 점수: CF={rec.cf_score:.2f} | CBF={rec.cbf_score:.2f} | MF={rec.mf_score:.2f}")
    print(f"     신뢰도: {rec.confidence:.0%}")
    if rec.explanation:
        print(f"     추천 이유: {rec.explanation}")


def demo_personalized_recommendation(recommender, user_profiles):
    """개인화 추천 데모"""
    print_header("개인화된 상품 추천 (Personalized Recommendations)")

    for user_id in list(user_profiles.keys())[:3]:
        profile = user_profiles[user_id]

        print(f"\n{'─' * 50}")
        print(f"👤 사용자 ID: {user_id}")
        print(f"   피부 타입: {profile.get('skin_type', 'N/A')}")
        print(f"   연령대: {profile.get('age_group', 'N/A')}")
        print(f"   피부 고민: {', '.join(profile.get('skin_concerns', [])) or 'N/A'}")
        print(f"   평균 평점: {profile.get('avg_rating', 0):.2f}")
        print(f"   리뷰 수: {len(profile.get('reviewed_products', []))}")

        recommendations = recommender.recommend(user_id, top_n=3)

        print(f"\n   📦 추천 상품:")
        for i, rec in enumerate(recommendations, 1):
            print_recommendation(rec, i)


def demo_similar_products(recommender, product_info):
    """유사 상품 추천 데모"""
    print_header("유사 상품 추천 (Similar Products)")

    sample_products = list(product_info.keys())[:2]

    for product_name in sample_products:
        print(f"\n{'─' * 50}")
        print(f"📦 기준 상품: {product_name}")

        info = product_info[product_name]
        print(f"   평점: {info.get('avg_rating', 0):.2f} | 리뷰 수: {info.get('review_count', 0)}")

        similar = recommender.recommend_similar(product_name, top_n=3)

        print(f"\n   🔗 유사한 상품:")
        for i, rec in enumerate(similar, 1):
            print(f"     {i}. {rec.product_name}")
            print(f"        유사도: {rec.final_score:.3f}")


def demo_trending_products(recommender):
    """인기 상품 추천 데모"""
    print_header("인기/트렌딩 상품 (Trending Products)")

    trending = recommender.recommend_trending(top_n=5)

    for i, rec in enumerate(trending, 1):
        print(f"\n  {i}. {rec.product_name}")
        print(f"     {rec.explanation}")


def demo_recommendation_explanation(recommender, user_profiles, product_info):
    """추천 설명 데모"""
    print_header("추천 이유 상세 분석 (Explainable Recommendations)")

    user_id = list(user_profiles.keys())[0]
    product_name = list(product_info.keys())[0]

    explanation = recommender.explain_recommendation(user_id, product_name)

    print(f"\n👤 사용자: {user_id}")
    print(f"📦 상품: {product_name}")
    print(f"\n📊 분석 결과:")
    print(f"   최종 점수: {explanation['final_score']:.2f}/5.00")

    print(f"\n   사용자 상황:")
    ctx = explanation['user_context']
    print(f"   - 구매 이력: {ctx['n_purchases']}건")
    print(f"   - 콜드 스타트: {'예' if ctx['is_cold_start'] else '아니오'}")
    print(f"   - 활성 사용자: {'예' if ctx['is_active'] else '아니오'}")

    print(f"\n   조정된 가중치:")
    weights = explanation['adjusted_weights']
    print(f"   - 협업 필터링: {weights['cf']:.0%}")
    print(f"   - 콘텐츠 기반: {weights['cbf']:.0%}")
    print(f"   - 행렬 분해: {weights['mf']:.0%}")

    print(f"\n   모델별 점수:")
    scores = explanation['model_scores']
    print(f"   - CF: {scores['collaborative_filtering']['score']:.2f} (기여도: {scores['collaborative_filtering']['contribution']:.3f})")
    print(f"   - CBF: {scores['content_based']['score']:.2f} (기여도: {scores['content_based']['contribution']:.3f})")
    print(f"   - MF: {scores['matrix_factorization']['score']:.2f} (기여도: {scores['matrix_factorization']['contribution']:.3f})")

    print(f"\n   💡 추천 이유: {explanation['recommendation_reason']}")


def demo_cold_start_user(recommender, user_profiles, product_info):
    """신규 사용자 (콜드 스타트) 추천 데모"""
    print_header("신규 사용자 추천 (Cold Start)")

    # 존재하지 않는 사용자 ID로 콜드 스타트 시뮬레이션
    new_user_id = 9999999

    print(f"\n👤 신규 사용자 ID: {new_user_id} (구매 이력 없음)")

    recommendations = recommender.recommend(new_user_id, top_n=3)

    print(f"\n   📦 추천 상품 (인기 기반):")
    for i, rec in enumerate(recommendations, 1):
        print_recommendation(rec, i)

    print(f"\n   ℹ️ 신규 사용자에게는 인기 상품을 우선 추천합니다.")
    print(f"      구매 이력이 쌓이면 개인화된 추천이 제공됩니다.")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print(" 🛍️  개인화된 하이브리드 상품 추천 시스템")
    print(" 📊  Personalized Hybrid Recommendation System")
    print("=" * 60)

    # 1. 데이터 로드 및 전처리
    print_header("데이터 로드 및 전처리")

    processor = DataProcessor()

    # 샘플 데이터 파일 확인
    sample_file = os.path.join(project_root, 'data', 'sample_reviews.csv')

    if os.path.exists(sample_file):
        print(f"📂 데이터 파일 로드: {sample_file}")
        df = processor.load_data(sample_file)
    else:
        print("📂 샘플 데이터 생성 중...")
        df = create_sample_dataframe()

    processed_df = processor.process_data(df)
    matrix, user_map, product_map = processor.create_user_item_matrix()

    print(f"\n   ✓ 총 {len(processed_df)} 건의 리뷰 데이터 처리 완료")
    print(f"   ✓ {len(processor.user_profiles)} 명의 사용자 프로필 구축")
    print(f"   ✓ {len(processor.product_info)} 개의 상품 정보 구축")

    # 2. 특성 추출기 초기화
    print_header("특성 추출기 학습")

    extractor = FeatureExtractor(tfidf_max_features=50)
    extractor.fit(processed_df)

    print(f"   ✓ TF-IDF 벡터라이저 학습 완료")
    print(f"   ✓ 사용자/상품 특성 추출 준비 완료")

    # 3. 하이브리드 추천 시스템 학습
    print_header("하이브리드 추천 시스템 학습")

    recommender = HybridRecommender(
        cf_weight=0.35,
        cbf_weight=0.35,
        mf_weight=0.30,
        cold_start_threshold=3
    )

    recommender.fit(
        processor.user_profiles,
        processor.product_info,
        matrix,
        processed_df,
        extractor,
        verbose=True
    )

    # 4. 데모 실행
    demo_personalized_recommendation(recommender, processor.user_profiles)
    demo_similar_products(recommender, processor.product_info)
    demo_trending_products(recommender)
    demo_recommendation_explanation(recommender, processor.user_profiles, processor.product_info)
    demo_cold_start_user(recommender, processor.user_profiles, processor.product_info)

    # 5. 모델 성능 요약
    print_header("모델 성능 요약")

    performance = recommender.get_model_performance()

    print(f"\n   📈 모델 구성:")
    print(f"   - 사용자 수: {performance['n_users']}")
    print(f"   - 상품 수: {performance['n_products']}")
    print(f"\n   ⚖️ 가중치 설정:")
    print(f"   - 협업 필터링 (CF): {performance['weights']['cf']:.0%}")
    print(f"   - 콘텐츠 기반 (CBF): {performance['weights']['cbf']:.0%}")
    print(f"   - 행렬 분해 (MF): {performance['weights']['mf']:.0%}")
    print(f"\n   🔧 모델 파라미터:")
    print(f"   - CF 이웃 수: {performance['cf_model']['n_neighbors']}")
    print(f"   - MF 잠재 요인 수: {performance['mf_model']['n_factors']}")
    print(f"   - 콜드 스타트 기준: {performance['cold_start_threshold']}건 미만")

    print("\n" + "=" * 60)
    print(" ✅ 추천 시스템 데모 완료!")
    print("=" * 60 + "\n")

    return recommender, processor


if __name__ == "__main__":
    recommender, processor = main()
