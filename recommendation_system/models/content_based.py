"""
콘텐츠 기반 필터링 모델
- 사용자 프로필 기반 상품 추천
- 상품 속성 기반 유사 상품 추천
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import sys
sys.path.insert(0, '..')


@dataclass
class CBFRecommendation:
    """콘텐츠 기반 추천 결과"""
    product_name: str
    similarity_score: float
    matching_attributes: Dict[str, any]
    explanation: str


class ContentBasedFiltering:
    """
    콘텐츠 기반 필터링 추천 모델

    주요 기능:
    - 사용자 프로필(피부 타입, 연령대, 피부 고민)과 상품 특성 매칭
    - 텍스트 기반 상품 유사도 계산
    - 설문 응답 기반 상품 매칭
    """

    def __init__(self, feature_extractor=None):
        """
        Args:
            feature_extractor: FeatureExtractor 인스턴스
        """
        self.feature_extractor = feature_extractor
        self.user_profiles = None
        self.product_info = None
        self.processed_df = None
        self.product_feature_matrix = None
        self.product_names = None

    def fit(
        self,
        user_profiles: Dict,
        product_info: Dict,
        processed_df: pd.DataFrame,
        feature_extractor=None
    ):
        """
        모델 학습

        Args:
            user_profiles: 사용자 프로필 딕셔너리
            product_info: 상품 정보 딕셔너리
            processed_df: 전처리된 DataFrame
            feature_extractor: 특성 추출기 (선택)
        """
        self.user_profiles = user_profiles
        self.product_info = product_info
        self.processed_df = processed_df

        if feature_extractor:
            self.feature_extractor = feature_extractor

        # 상품 특성 행렬 구축
        if self.feature_extractor:
            self.product_feature_matrix, self.product_names = \
                self.feature_extractor.build_product_feature_matrix(
                    product_info,
                    processed_df
                )
        else:
            # 간단한 특성 행렬 구축
            self._build_simple_product_features()

        print(f"Content-based filtering fitted: {len(user_profiles)} users, {len(product_info)} products")

    def _build_simple_product_features(self):
        """간단한 상품 특성 행렬 구축 (feature_extractor 없이)"""
        self.product_names = list(self.product_info.keys())
        features_list = []

        # 피부 타입 및 연령대 인코딩
        skin_types = ['dry', 'oily', 'combination', 'normal', 'sensitive']
        age_groups = ['10s', '20s', '30s', '40s', '50s', '50+', '60s', '60+']

        for product_name in self.product_names:
            info = self.product_info[product_name]

            # 기본 특성
            features = [
                np.log1p(info.get('review_count', 0)),
                info.get('avg_rating', 3.0) / 5.0,
                np.log1p(info.get('total_recommends', 0))
            ]

            # 피부 타입 벡터
            skin_vec = [0] * len(skin_types)
            for i, st in enumerate(info.get('primary_skin_types', [])):
                if st in skin_types:
                    skin_vec[skin_types.index(st)] = (3 - i) / 3
            features.extend(skin_vec)

            # 연령대 벡터
            age_vec = [0] * len(age_groups)
            for i, ag in enumerate(info.get('primary_age_groups', [])):
                if ag in age_groups:
                    age_vec[age_groups.index(ag)] = (3 - i) / 3
            features.extend(age_vec)

            features_list.append(features)

        self.product_feature_matrix = np.array(features_list)

    def _calculate_profile_match_score(
        self,
        user_profile: Dict,
        product_info: Dict
    ) -> Tuple[float, Dict]:
        """
        사용자 프로필과 상품 간 매칭 점수 계산

        Args:
            user_profile: 사용자 프로필
            product_info: 상품 정보

        Returns:
            (매칭 점수, 매칭된 속성)
        """
        score = 0.0
        max_score = 0.0
        matching_attrs = {}

        # 피부 타입 매칭 (가중치: 3)
        user_skin_type = user_profile.get('skin_type')
        product_skin_types = product_info.get('primary_skin_types', [])

        max_score += 3.0
        if user_skin_type and user_skin_type in product_skin_types:
            position = product_skin_types.index(user_skin_type)
            skin_score = 3.0 * (3 - position) / 3  # 상위일수록 높은 점수
            score += skin_score
            matching_attrs['skin_type'] = {
                'user': user_skin_type,
                'product_rank': position + 1,
                'score': skin_score
            }

        # 연령대 매칭 (가중치: 2)
        user_age_group = user_profile.get('age_group')
        product_age_groups = product_info.get('primary_age_groups', [])

        max_score += 2.0
        if user_age_group and user_age_group in product_age_groups:
            position = product_age_groups.index(user_age_group)
            age_score = 2.0 * (3 - position) / 3
            score += age_score
            matching_attrs['age_group'] = {
                'user': user_age_group,
                'product_rank': position + 1,
                'score': age_score
            }

        # 피부 고민 매칭 (가중치: 2)
        user_concerns = set(user_profile.get('skin_concerns', []))
        if user_concerns:
            # 상품 리뷰에서 피부 고민 관련 키워드 매칭
            concern_keywords = {
                'wrinkle': ['주름', '탄력', '안티에이징'],
                'pore': ['모공', '피지'],
                'whitening': ['미백', '톤업', '브라이트닝'],
                'sensitivity': ['민감', '순한', '저자극'],
                'dryness': ['건조', '보습', '수분'],
                'trouble': ['트러블', '여드름', '진정']
            }

            concern_score = 0.0
            matched_concerns = []

            # 상품 설문 응답에서 매칭 확인
            survey_summary = product_info.get('survey_summary', {})
            for concern in user_concerns:
                if concern in concern_keywords:
                    # 설문 응답에서 관련 키워드 찾기
                    for question, responses in survey_summary.items():
                        for response in responses.keys():
                            if any(kw in response for kw in concern_keywords[concern]):
                                concern_score += 0.5
                                matched_concerns.append(concern)
                                break

            max_score += 2.0
            score += min(concern_score, 2.0)
            if matched_concerns:
                matching_attrs['skin_concerns'] = {
                    'user': list(user_concerns),
                    'matched': list(set(matched_concerns)),
                    'score': min(concern_score, 2.0)
                }

        # 상품 평점 가산점 (가중치: 1)
        avg_rating = product_info.get('avg_rating', 3.0)
        max_score += 1.0
        rating_bonus = (avg_rating - 3.0) / 2.0  # 3점 기준, -1 ~ 1 범위
        score += max(0, rating_bonus)
        matching_attrs['rating_bonus'] = rating_bonus

        # 정규화
        final_score = score / max_score if max_score > 0 else 0

        return final_score, matching_attrs

    def _generate_explanation(
        self,
        matching_attrs: Dict,
        product_info: Dict
    ) -> str:
        """추천 설명 생성"""
        explanations = []

        if 'skin_type' in matching_attrs:
            skin_info = matching_attrs['skin_type']
            explanations.append(
                f"'{skin_info['user']}' 피부 타입에 적합한 상품입니다"
            )

        if 'age_group' in matching_attrs:
            age_info = matching_attrs['age_group']
            age_kr = {
                '10s': '10대', '20s': '20대', '30s': '30대',
                '40s': '40대', '50s': '50대', '50+': '50대 이상'
            }
            explanations.append(
                f"{age_kr.get(age_info['user'], age_info['user'])} 고객님들이 선호하는 상품입니다"
            )

        if 'skin_concerns' in matching_attrs:
            matched = matching_attrs['skin_concerns']['matched']
            concern_kr = {
                'wrinkle': '주름', 'pore': '모공', 'whitening': '미백',
                'sensitivity': '민감성', 'dryness': '건조', 'trouble': '트러블'
            }
            concerns_str = ', '.join([concern_kr.get(c, c) for c in matched[:2]])
            explanations.append(f"{concerns_str} 고민에 도움이 됩니다")

        avg_rating = product_info.get('avg_rating', 0)
        if avg_rating >= 4.5:
            explanations.append(f"평점 {avg_rating:.1f}점의 인기 상품입니다")

        return ' | '.join(explanations) if explanations else "추천 상품입니다"

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_purchased: bool = True
    ) -> List[CBFRecommendation]:
        """
        사용자에게 상품 추천

        Args:
            user_id: 사용자 ID
            top_n: 추천 상품 수
            exclude_purchased: 구매한 상품 제외 여부

        Returns:
            추천 결과 리스트
        """
        user_profile = self.user_profiles.get(user_id)

        if not user_profile:
            # 콜드 스타트: 인기 상품 추천
            return self._recommend_popular(top_n)

        # 구매한 상품 목록
        purchased = set()
        if exclude_purchased:
            for p in user_profile.get('reviewed_products', []):
                purchased.add(p['product_name'])

        recommendations = []

        for product_name, product_info in self.product_info.items():
            if product_name in purchased:
                continue

            # 프로필 매칭 점수 계산
            match_score, matching_attrs = self._calculate_profile_match_score(
                user_profile,
                product_info
            )

            # 설명 생성
            explanation = self._generate_explanation(matching_attrs, product_info)

            recommendations.append(CBFRecommendation(
                product_name=product_name,
                similarity_score=match_score,
                matching_attributes=matching_attrs,
                explanation=explanation
            ))

        # 점수 기준 정렬
        recommendations.sort(key=lambda x: x.similarity_score, reverse=True)

        return recommendations[:top_n]

    def _recommend_popular(self, top_n: int) -> List[CBFRecommendation]:
        """인기 상품 추천 (콜드 스타트)"""
        # 평점 기준 정렬
        sorted_products = sorted(
            self.product_info.items(),
            key=lambda x: (x[1].get('avg_rating', 0), x[1].get('review_count', 0)),
            reverse=True
        )

        recommendations = []
        for product_name, info in sorted_products[:top_n]:
            recommendations.append(CBFRecommendation(
                product_name=product_name,
                similarity_score=info.get('avg_rating', 3.0) / 5.0,
                matching_attributes={'popular': True},
                explanation=f"평점 {info.get('avg_rating', 0):.1f}점의 인기 상품입니다"
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
        if product_name not in self.product_names:
            return []

        product_idx = self.product_names.index(product_name)

        # 코사인 유사도 계산
        product_vec = self.product_feature_matrix[product_idx].reshape(1, -1)
        similarities = cosine_similarity(product_vec, self.product_feature_matrix)[0]

        # 자기 자신 제외하고 정렬
        similar_indices = np.argsort(similarities)[::-1]

        result = []
        for idx in similar_indices:
            if idx == product_idx:
                continue
            if len(result) >= top_n:
                break
            result.append((self.product_names[idx], similarities[idx]))

        return result

    def recommend_for_product(
        self,
        product_name: str,
        user_id: int = None,
        top_n: int = 5
    ) -> List[CBFRecommendation]:
        """
        특정 상품을 본 사용자에게 유사 상품 추천

        Args:
            product_name: 현재 보고 있는 상품
            user_id: 사용자 ID (선택)
            top_n: 추천 상품 수

        Returns:
            추천 결과 리스트
        """
        similar_products = self.get_similar_products(product_name, top_n * 2)

        recommendations = []

        for similar_name, similarity in similar_products:
            if len(recommendations) >= top_n:
                break

            product_info = self.product_info.get(similar_name, {})

            # 사용자가 있으면 프로필 매칭도 고려
            if user_id and user_id in self.user_profiles:
                profile_score, matching_attrs = self._calculate_profile_match_score(
                    self.user_profiles[user_id],
                    product_info
                )
                combined_score = (similarity + profile_score) / 2
            else:
                combined_score = similarity
                matching_attrs = {'similarity': similarity}

            recommendations.append(CBFRecommendation(
                product_name=similar_name,
                similarity_score=combined_score,
                matching_attributes=matching_attrs,
                explanation=f"'{product_name[:20]}...'와 유사한 상품입니다"
            ))

        return recommendations

    def explain_recommendation(
        self,
        user_id: int,
        product_name: str
    ) -> Dict:
        """
        추천 이유 상세 설명

        Args:
            user_id: 사용자 ID
            product_name: 상품명

        Returns:
            상세 설명 딕셔너리
        """
        user_profile = self.user_profiles.get(user_id)
        product_info = self.product_info.get(product_name)

        if not user_profile or not product_info:
            return {'error': 'User or product not found'}

        score, matching_attrs = self._calculate_profile_match_score(
            user_profile,
            product_info
        )

        return {
            'user_id': user_id,
            'product_name': product_name,
            'overall_score': score,
            'user_profile': {
                'skin_type': user_profile.get('skin_type'),
                'age_group': user_profile.get('age_group'),
                'skin_concerns': user_profile.get('skin_concerns', [])
            },
            'product_attributes': {
                'primary_skin_types': product_info.get('primary_skin_types', []),
                'primary_age_groups': product_info.get('primary_age_groups', []),
                'avg_rating': product_info.get('avg_rating', 0),
                'review_count': product_info.get('review_count', 0)
            },
            'matching_details': matching_attrs,
            'explanation': self._generate_explanation(matching_attrs, product_info)
        }


if __name__ == "__main__":
    from utils.data_processor import DataProcessor, create_sample_dataframe
    from utils.feature_extractor import FeatureExtractor

    # 데이터 준비
    processor = DataProcessor()
    df = create_sample_dataframe()
    processed_df = processor.process_data(df)

    # 특성 추출기
    extractor = FeatureExtractor(tfidf_max_features=50)
    extractor.fit(processed_df)

    # 콘텐츠 기반 모델 학습
    cbf_model = ContentBasedFiltering()
    cbf_model.fit(
        processor.user_profiles,
        processor.product_info,
        processed_df,
        extractor
    )

    # 추천
    for user_id in list(processor.user_profiles.keys())[:2]:
        print(f"\n=== User {user_id} 콘텐츠 기반 추천 ===")
        profile = processor.user_profiles[user_id]
        print(f"  프로필: {profile.get('skin_type')}, {profile.get('age_group')}")

        recommendations = cbf_model.recommend(user_id, top_n=3)
        for rec in recommendations:
            print(f"  {rec.product_name[:30]}...")
            print(f"    점수: {rec.similarity_score:.3f}")
            print(f"    설명: {rec.explanation}")
