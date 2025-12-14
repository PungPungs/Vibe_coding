"""
하이브리드 추천 시스템
- 협업 필터링 + 콘텐츠 기반 + 행렬 분해 통합
- 가중치 기반 앙상블
- 상황별 적응형 추천
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
sys.path.insert(0, '..')

from .collaborative_filtering import CollaborativeFiltering, CFRecommendation
from .content_based import ContentBasedFiltering, CBFRecommendation
from .matrix_factorization import MatrixFactorization, MFRecommendation


class RecommendationType(Enum):
    """추천 유형"""
    PERSONALIZED = "personalized"       # 개인화 추천
    COLD_START = "cold_start"          # 콜드 스타트 (신규 사용자)
    SIMILAR_PRODUCTS = "similar"       # 유사 상품
    TRENDING = "trending"              # 인기/트렌딩
    CONTEXT_AWARE = "context_aware"    # 상황 인식 추천


@dataclass
class HybridRecommendation:
    """하이브리드 추천 결과"""
    product_name: str
    final_score: float
    cf_score: float
    cbf_score: float
    mf_score: float
    explanation: str
    recommendation_type: RecommendationType
    confidence: float
    contributing_factors: Dict[str, float] = field(default_factory=dict)


class HybridRecommender:
    """
    하이브리드 추천 시스템

    주요 기능:
    - 다중 추천 알고리즘 통합
    - 적응형 가중치 조정
    - 콜드 스타트 처리
    - 설명 가능한 추천
    """

    def __init__(
        self,
        cf_weight: float = 0.35,
        cbf_weight: float = 0.35,
        mf_weight: float = 0.30,
        cold_start_threshold: int = 3
    ):
        """
        Args:
            cf_weight: 협업 필터링 가중치
            cbf_weight: 콘텐츠 기반 가중치
            mf_weight: 행렬 분해 가중치
            cold_start_threshold: 콜드 스타트 판단 기준 (최소 리뷰 수)
        """
        self.cf_weight = cf_weight
        self.cbf_weight = cbf_weight
        self.mf_weight = mf_weight
        self.cold_start_threshold = cold_start_threshold

        # 개별 모델
        self.cf_model = None
        self.cbf_model = None
        self.mf_model = None

        # 데이터
        self.user_profiles = None
        self.product_info = None
        self.user_item_matrix = None
        self.processed_df = None

        self.is_fitted = False

    def fit(
        self,
        user_profiles: Dict,
        product_info: Dict,
        user_item_matrix: pd.DataFrame,
        processed_df: pd.DataFrame,
        feature_extractor=None,
        verbose: bool = True
    ):
        """
        전체 모델 학습

        Args:
            user_profiles: 사용자 프로필 딕셔너리
            product_info: 상품 정보 딕셔너리
            user_item_matrix: 사용자-상품 상호작용 매트릭스
            processed_df: 전처리된 DataFrame
            feature_extractor: 특성 추출기
            verbose: 학습 과정 출력 여부
        """
        self.user_profiles = user_profiles
        self.product_info = product_info
        self.user_item_matrix = user_item_matrix
        self.processed_df = processed_df

        if verbose:
            print("=" * 50)
            print("하이브리드 추천 시스템 학습 시작")
            print("=" * 50)

        # 1. 협업 필터링 모델
        if verbose:
            print("\n[1/3] 협업 필터링 모델 학습...")
        self.cf_model = CollaborativeFiltering(
            method='user',
            n_neighbors=min(20, len(user_profiles) - 1),
            similarity_threshold=0.0
        )
        self.cf_model.fit(user_item_matrix)

        # 2. 콘텐츠 기반 필터링 모델
        if verbose:
            print("\n[2/3] 콘텐츠 기반 필터링 모델 학습...")
        self.cbf_model = ContentBasedFiltering(feature_extractor)
        self.cbf_model.fit(user_profiles, product_info, processed_df, feature_extractor)

        # 3. 행렬 분해 모델
        if verbose:
            print("\n[3/3] 행렬 분해 모델 학습...")
        n_factors = min(20, min(len(user_profiles), len(product_info)) - 1)
        self.mf_model = MatrixFactorization(
            n_factors=max(2, n_factors),
            n_epochs=50,
            method='als'
        )
        self.mf_model.fit(user_item_matrix, verbose=False)

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 50)
            print("하이브리드 추천 시스템 학습 완료!")
            print(f"  - 사용자 수: {len(user_profiles)}")
            print(f"  - 상품 수: {len(product_info)}")
            print(f"  - CF 가중치: {self.cf_weight:.2f}")
            print(f"  - CBF 가중치: {self.cbf_weight:.2f}")
            print(f"  - MF 가중치: {self.mf_weight:.2f}")
            print("=" * 50)

    def _get_user_context(self, user_id: int) -> Dict[str, Any]:
        """사용자 상황 정보 수집"""
        user_profile = self.user_profiles.get(user_id, {})

        # 구매 이력 수
        n_purchases = len(user_profile.get('reviewed_products', []))

        # 콜드 스타트 여부
        is_cold_start = n_purchases < self.cold_start_threshold

        # 활성 사용자 여부
        is_active = n_purchases >= 5

        return {
            'n_purchases': n_purchases,
            'is_cold_start': is_cold_start,
            'is_active': is_active,
            'skin_type': user_profile.get('skin_type'),
            'age_group': user_profile.get('age_group'),
            'avg_rating': user_profile.get('avg_rating', 3.0)
        }

    def _adjust_weights(self, user_context: Dict) -> Tuple[float, float, float]:
        """
        사용자 상황에 따른 가중치 조정

        Args:
            user_context: 사용자 상황 정보

        Returns:
            (cf_weight, cbf_weight, mf_weight)
        """
        cf_w = self.cf_weight
        cbf_w = self.cbf_weight
        mf_w = self.mf_weight

        if user_context['is_cold_start']:
            # 콜드 스타트: 콘텐츠 기반 강화, 협업 필터링 약화
            cf_w = 0.15
            cbf_w = 0.60
            mf_w = 0.25

        elif user_context['is_active']:
            # 활성 사용자: 협업 필터링 및 행렬 분해 강화
            cf_w = 0.40
            cbf_w = 0.25
            mf_w = 0.35

        # 정규화
        total = cf_w + cbf_w + mf_w
        return cf_w / total, cbf_w / total, mf_w / total

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """점수 정규화 (0~1 범위)"""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_purchased: bool = True,
        context: Dict = None
    ) -> List[HybridRecommendation]:
        """
        개인화된 상품 추천

        Args:
            user_id: 사용자 ID
            top_n: 추천 상품 수
            exclude_purchased: 구매한 상품 제외 여부
            context: 추가 상황 정보 (선택)

        Returns:
            하이브리드 추천 결과 리스트
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 사용자 상황 분석
        user_context = self._get_user_context(user_id)
        if context:
            user_context.update(context)

        # 가중치 조정
        cf_w, cbf_w, mf_w = self._adjust_weights(user_context)

        # 추천 유형 결정
        if user_context['is_cold_start']:
            rec_type = RecommendationType.COLD_START
        else:
            rec_type = RecommendationType.PERSONALIZED

        # 각 모델별 추천 결과 수집
        all_products = list(self.product_info.keys())

        # 구매한 상품
        purchased = set()
        if exclude_purchased and user_id in self.user_profiles:
            for p in self.user_profiles[user_id].get('reviewed_products', []):
                purchased.add(p['product_name'])

        # 점수 수집
        product_scores = {}

        for product_name in all_products:
            if product_name in purchased:
                continue

            # CF 점수
            cf_score, cf_conf = self.cf_model.predict(user_id, product_name)
            cf_score_norm = cf_score / 5.0

            # CBF 점수
            cbf_result = self._get_cbf_score(user_id, product_name)
            cbf_score_norm = cbf_result['score']

            # MF 점수
            mf_score = self.mf_model.predict(user_id, product_name)
            mf_score_norm = mf_score / 5.0

            # 가중 평균
            final_score = (
                cf_w * cf_score_norm +
                cbf_w * cbf_score_norm +
                mf_w * mf_score_norm
            )

            # 신뢰도 계산
            confidence = (cf_conf + cbf_result.get('confidence', 0.5)) / 2

            product_scores[product_name] = {
                'final_score': final_score,
                'cf_score': cf_score,
                'cbf_score': cbf_score_norm * 5,
                'mf_score': mf_score,
                'explanation': cbf_result.get('explanation', ''),
                'confidence': confidence,
                'contributing_factors': {
                    'cf_contribution': cf_w * cf_score_norm,
                    'cbf_contribution': cbf_w * cbf_score_norm,
                    'mf_contribution': mf_w * mf_score_norm
                }
            }

        # 점수 기준 정렬
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: (x[1]['final_score'], x[1]['confidence']),
            reverse=True
        )

        # 결과 생성
        recommendations = []
        for product_name, scores in sorted_products[:top_n]:
            recommendations.append(HybridRecommendation(
                product_name=product_name,
                final_score=scores['final_score'] * 5,  # 1-5 스케일로 변환
                cf_score=scores['cf_score'],
                cbf_score=scores['cbf_score'],
                mf_score=scores['mf_score'],
                explanation=scores['explanation'],
                recommendation_type=rec_type,
                confidence=scores['confidence'],
                contributing_factors=scores['contributing_factors']
            ))

        return recommendations

    def _get_cbf_score(self, user_id: int, product_name: str) -> Dict:
        """CBF 점수 계산"""
        if user_id not in self.user_profiles:
            return {'score': 0.5, 'confidence': 0.3, 'explanation': ''}

        user_profile = self.user_profiles[user_id]
        product_info = self.product_info.get(product_name, {})

        if not product_info:
            return {'score': 0.5, 'confidence': 0.3, 'explanation': ''}

        # 프로필 매칭 점수 계산
        score, matching_attrs = self.cbf_model._calculate_profile_match_score(
            user_profile,
            product_info
        )

        # 설명 생성
        explanation = self.cbf_model._generate_explanation(matching_attrs, product_info)

        # 신뢰도: 매칭된 속성 수에 비례
        n_matched = len([k for k in matching_attrs.keys() if k != 'rating_bonus'])
        confidence = min(n_matched / 3, 1.0)

        return {
            'score': score,
            'confidence': confidence,
            'explanation': explanation
        }

    def recommend_similar(
        self,
        product_name: str,
        user_id: int = None,
        top_n: int = 5
    ) -> List[HybridRecommendation]:
        """
        유사 상품 추천

        Args:
            product_name: 기준 상품명
            user_id: 사용자 ID (선택, 있으면 개인화 반영)
            top_n: 추천 상품 수

        Returns:
            추천 결과 리스트
        """
        # 각 모델의 유사 상품
        cf_similar = dict(self.cf_model.get_similar_products(product_name, top_n * 2))
        cbf_similar = dict(self.cbf_model.get_similar_products(product_name, top_n * 2))
        mf_similar = dict(self.mf_model.get_similar_products(product_name, top_n * 2))

        # 모든 유사 상품 후보
        all_similar = set(cf_similar.keys()) | set(cbf_similar.keys()) | set(mf_similar.keys())

        product_scores = {}

        for similar_name in all_similar:
            cf_sim = cf_similar.get(similar_name, 0)
            cbf_sim = cbf_similar.get(similar_name, 0)
            mf_sim = mf_similar.get(similar_name, 0)

            # 가중 평균
            final_sim = (
                self.cf_weight * cf_sim +
                self.cbf_weight * cbf_sim +
                self.mf_weight * mf_sim
            )

            # 사용자가 있으면 개인화 점수 추가
            if user_id and user_id in self.user_profiles:
                personal_score = self._get_cbf_score(user_id, similar_name)['score']
                final_sim = final_sim * 0.7 + personal_score * 0.3

            product_scores[similar_name] = {
                'final_score': final_sim,
                'cf_score': cf_sim,
                'cbf_score': cbf_sim,
                'mf_score': mf_sim
            }

        # 정렬
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )

        recommendations = []
        for similar_name, scores in sorted_products[:top_n]:
            recommendations.append(HybridRecommendation(
                product_name=similar_name,
                final_score=scores['final_score'],
                cf_score=scores['cf_score'],
                cbf_score=scores['cbf_score'],
                mf_score=scores['mf_score'],
                explanation=f"'{product_name[:20]}...'와 유사한 상품입니다",
                recommendation_type=RecommendationType.SIMILAR_PRODUCTS,
                confidence=0.7,
                contributing_factors={}
            ))

        return recommendations

    def recommend_trending(self, top_n: int = 10) -> List[HybridRecommendation]:
        """
        트렌딩/인기 상품 추천

        Args:
            top_n: 추천 상품 수

        Returns:
            추천 결과 리스트
        """
        # 상품별 인기도 점수
        product_scores = {}

        for product_name, info in self.product_info.items():
            # 리뷰 수, 평점, 추천 수 기반 인기도
            review_score = np.log1p(info.get('review_count', 0))
            rating_score = info.get('avg_rating', 3.0) / 5.0
            recommend_score = np.log1p(info.get('total_recommends', 0))

            popularity = (
                review_score * 0.3 +
                rating_score * 0.5 +
                recommend_score * 0.2
            )

            product_scores[product_name] = {
                'popularity': popularity,
                'avg_rating': info.get('avg_rating', 3.0),
                'review_count': info.get('review_count', 0)
            }

        # 정렬
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1]['popularity'],
            reverse=True
        )

        recommendations = []
        for product_name, scores in sorted_products[:top_n]:
            recommendations.append(HybridRecommendation(
                product_name=product_name,
                final_score=scores['popularity'] * 5,
                cf_score=0,
                cbf_score=0,
                mf_score=0,
                explanation=f"평점 {scores['avg_rating']:.1f}점, {scores['review_count']}개 리뷰",
                recommendation_type=RecommendationType.TRENDING,
                confidence=0.8,
                contributing_factors={}
            ))

        return recommendations

    def explain_recommendation(
        self,
        user_id: int,
        product_name: str
    ) -> Dict[str, Any]:
        """
        추천 이유 상세 설명

        Args:
            user_id: 사용자 ID
            product_name: 상품명

        Returns:
            상세 설명 딕셔너리
        """
        user_context = self._get_user_context(user_id)
        cf_w, cbf_w, mf_w = self._adjust_weights(user_context)

        # 각 모델별 점수
        cf_score, cf_conf = self.cf_model.predict(user_id, product_name)
        cbf_result = self._get_cbf_score(user_id, product_name)
        mf_score = self.mf_model.predict(user_id, product_name)

        # CBF 상세 설명
        cbf_explanation = self.cbf_model.explain_recommendation(user_id, product_name)

        return {
            'user_id': user_id,
            'product_name': product_name,
            'user_context': user_context,
            'adjusted_weights': {
                'cf': cf_w,
                'cbf': cbf_w,
                'mf': mf_w
            },
            'model_scores': {
                'collaborative_filtering': {
                    'score': cf_score,
                    'confidence': cf_conf,
                    'contribution': cf_w * (cf_score / 5.0)
                },
                'content_based': {
                    'score': cbf_result['score'] * 5,
                    'confidence': cbf_result['confidence'],
                    'contribution': cbf_w * cbf_result['score'],
                    'explanation': cbf_result['explanation']
                },
                'matrix_factorization': {
                    'score': mf_score,
                    'contribution': mf_w * (mf_score / 5.0)
                }
            },
            'final_score': (
                cf_w * (cf_score / 5.0) +
                cbf_w * cbf_result['score'] +
                mf_w * (mf_score / 5.0)
            ) * 5,
            'profile_matching': cbf_explanation.get('matching_details', {}),
            'recommendation_reason': self._generate_recommendation_reason(
                user_context,
                cf_score,
                cbf_result,
                mf_score
            )
        }

    def _generate_recommendation_reason(
        self,
        user_context: Dict,
        cf_score: float,
        cbf_result: Dict,
        mf_score: float
    ) -> str:
        """추천 이유 텍스트 생성"""
        reasons = []

        # 협업 필터링 이유
        if cf_score >= 4.0:
            reasons.append("비슷한 취향의 고객들이 좋아한 상품입니다")

        # 콘텐츠 기반 이유
        if cbf_result.get('explanation'):
            reasons.append(cbf_result['explanation'])

        # 행렬 분해 이유
        if mf_score >= 4.0:
            reasons.append("고객님의 구매 패턴에 잘 맞는 상품입니다")

        # 콜드 스타트인 경우
        if user_context.get('is_cold_start'):
            reasons.append("프로필 정보를 바탕으로 추천드립니다")

        return ' | '.join(reasons) if reasons else "종합적인 분석을 통해 추천드립니다"

    def get_model_performance(self) -> Dict[str, Any]:
        """모델 성능 요약"""
        return {
            'n_users': len(self.user_profiles) if self.user_profiles else 0,
            'n_products': len(self.product_info) if self.product_info else 0,
            'weights': {
                'cf': self.cf_weight,
                'cbf': self.cbf_weight,
                'mf': self.mf_weight
            },
            'cold_start_threshold': self.cold_start_threshold,
            'cf_model': {
                'method': self.cf_model.method if self.cf_model else None,
                'n_neighbors': self.cf_model.n_neighbors if self.cf_model else None
            },
            'mf_model': {
                'n_factors': self.mf_model.n_factors if self.mf_model else None,
                'method': self.mf_model.method if self.mf_model else None
            }
        }


if __name__ == "__main__":
    from utils.data_processor import DataProcessor, create_sample_dataframe
    from utils.feature_extractor import FeatureExtractor

    print("=" * 60)
    print("하이브리드 추천 시스템 테스트")
    print("=" * 60)

    # 데이터 준비
    processor = DataProcessor()
    df = create_sample_dataframe()
    processed_df = processor.process_data(df)
    matrix, user_map, product_map = processor.create_user_item_matrix()

    # 특성 추출기
    extractor = FeatureExtractor(tfidf_max_features=50)
    extractor.fit(processed_df)

    # 하이브리드 모델 학습
    hybrid = HybridRecommender(
        cf_weight=0.35,
        cbf_weight=0.35,
        mf_weight=0.30
    )
    hybrid.fit(
        processor.user_profiles,
        processor.product_info,
        matrix,
        processed_df,
        extractor,
        verbose=True
    )

    # 추천 테스트
    for user_id in list(processor.user_profiles.keys())[:2]:
        print(f"\n{'=' * 50}")
        print(f"사용자 {user_id} 개인화 추천")
        print(f"{'=' * 50}")

        profile = processor.user_profiles[user_id]
        print(f"프로필: {profile.get('skin_type')}, {profile.get('age_group')}, 피부고민: {profile.get('skin_concerns', [])}")

        recommendations = hybrid.recommend(user_id, top_n=3)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec.product_name}")
            print(f"   종합 점수: {rec.final_score:.2f}")
            print(f"   CF: {rec.cf_score:.2f} | CBF: {rec.cbf_score:.2f} | MF: {rec.mf_score:.2f}")
            print(f"   설명: {rec.explanation}")
            print(f"   신뢰도: {rec.confidence:.2f}")

    # 유사 상품 추천
    sample_product = list(processor.product_info.keys())[0]
    print(f"\n{'=' * 50}")
    print(f"'{sample_product[:30]}...'와 유사한 상품")
    print(f"{'=' * 50}")

    similar_recs = hybrid.recommend_similar(sample_product, top_n=3)
    for rec in similar_recs:
        print(f"  - {rec.product_name[:40]}... (유사도: {rec.final_score:.3f})")

    # 트렌딩 상품
    print(f"\n{'=' * 50}")
    print("인기 상품")
    print(f"{'=' * 50}")

    trending = hybrid.recommend_trending(top_n=3)
    for rec in trending:
        print(f"  - {rec.product_name[:40]}...")
        print(f"    {rec.explanation}")

    # 추천 설명
    print(f"\n{'=' * 50}")
    print("추천 이유 상세 분석")
    print(f"{'=' * 50}")

    user_id = list(processor.user_profiles.keys())[0]
    product_name = list(processor.product_info.keys())[0]
    explanation = hybrid.explain_recommendation(user_id, product_name)
    print(f"사용자: {user_id}")
    print(f"상품: {product_name[:30]}...")
    print(f"최종 점수: {explanation['final_score']:.2f}")
    print(f"추천 이유: {explanation['recommendation_reason']}")
