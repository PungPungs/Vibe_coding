"""
특성 추출 모듈
- 사용자 특성 벡터화
- 상품 특성 벡터화
- 텍스트 임베딩 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from dataclasses import dataclass


@dataclass
class UserFeatures:
    """사용자 특성 데이터"""
    member_sn: int
    demographic_vector: np.ndarray  # 인구통계학적 특성
    behavioral_vector: np.ndarray   # 행동 특성
    preference_vector: np.ndarray   # 선호도 특성


@dataclass
class ProductFeatures:
    """상품 특성 데이터"""
    product_name: str
    attribute_vector: np.ndarray    # 속성 벡터
    review_embedding: np.ndarray    # 리뷰 임베딩
    survey_vector: np.ndarray       # 설문 응답 벡터


class FeatureExtractor:
    """
    특성 추출 클래스

    주요 기능:
    - 사용자 인구통계학적 특성 벡터화
    - 사용자 행동 패턴 특성 추출
    - 상품 속성 벡터화
    - 리뷰 텍스트 TF-IDF 벡터화
    """

    # 피부 타입 목록
    SKIN_TYPES = ['dry', 'oily', 'combination', 'normal', 'sensitive']

    # 연령대 목록
    AGE_GROUPS = ['10s', '20s', '30s', '40s', '50s', '50+', '60s', '60+']

    # 성별 목록
    GENDERS = ['male', 'female']

    # 피부 고민 목록
    SKIN_CONCERNS = [
        'wrinkle', 'elasticity', 'whitening', 'pore', 'trouble',
        'dead_skin', 'dryness', 'oiliness', 'sensitivity', 'blemish', 'dark_circle'
    ]

    def __init__(self, tfidf_max_features: int = 100):
        """
        Args:
            tfidf_max_features: TF-IDF 벡터의 최대 특성 수
        """
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

        # 인코더 초기화
        self._init_encoders()

    def _init_encoders(self):
        """원-핫 인코더 초기화"""
        self.skin_type_encoder = {st: idx for idx, st in enumerate(self.SKIN_TYPES)}
        self.age_group_encoder = {ag: idx for idx, ag in enumerate(self.AGE_GROUPS)}
        self.gender_encoder = {g: idx for idx, g in enumerate(self.GENDERS)}
        self.skin_concern_encoder = {sc: idx for idx, sc in enumerate(self.SKIN_CONCERNS)}

    def fit(self, processed_df: pd.DataFrame):
        """
        특성 추출기 학습

        Args:
            processed_df: 전처리된 DataFrame
        """
        # TF-IDF 학습
        review_texts = processed_df['cleaned_review'].fillna('').tolist()
        self.tfidf_vectorizer.fit(review_texts)
        self.is_fitted = True

        print(f"Feature extractor fitted with {len(review_texts)} reviews")

    def extract_user_demographic_features(self, user_profile: Dict) -> np.ndarray:
        """
        사용자 인구통계학적 특성 추출

        Args:
            user_profile: 사용자 프로필 딕셔너리

        Returns:
            특성 벡터 (numpy array)
        """
        # 피부 타입 원-핫 인코딩 (5차원)
        skin_type_vec = np.zeros(len(self.SKIN_TYPES))
        skin_type = user_profile.get('skin_type')
        if skin_type and skin_type in self.skin_type_encoder:
            skin_type_vec[self.skin_type_encoder[skin_type]] = 1

        # 연령대 원-핫 인코딩 (8차원)
        age_group_vec = np.zeros(len(self.AGE_GROUPS))
        age_group = user_profile.get('age_group')
        if age_group and age_group in self.age_group_encoder:
            age_group_vec[self.age_group_encoder[age_group]] = 1

        # 성별 원-핫 인코딩 (2차원)
        gender_vec = np.zeros(len(self.GENDERS))
        gender = user_profile.get('gender')
        if gender and gender in self.gender_encoder:
            gender_vec[self.gender_encoder[gender]] = 1

        # 피부 고민 멀티-핫 인코딩 (11차원)
        skin_concern_vec = np.zeros(len(self.SKIN_CONCERNS))
        skin_concerns = user_profile.get('skin_concerns', [])
        for concern in skin_concerns:
            if concern in self.skin_concern_encoder:
                skin_concern_vec[self.skin_concern_encoder[concern]] = 1

        # 전체 벡터 결합 (26차원)
        demographic_vector = np.concatenate([
            skin_type_vec,
            age_group_vec,
            gender_vec,
            skin_concern_vec
        ])

        return demographic_vector

    def extract_user_behavioral_features(self, user_profile: Dict) -> np.ndarray:
        """
        사용자 행동 특성 추출

        Args:
            user_profile: 사용자 프로필 딕셔너리

        Returns:
            행동 특성 벡터
        """
        reviewed_products = user_profile.get('reviewed_products', [])

        if not reviewed_products:
            return np.zeros(5)

        # 리뷰 수
        review_count = len(reviewed_products)

        # 평균 평점
        avg_rating = np.mean([p['scope'] for p in reviewed_products])

        # 평점 표준편차 (선호도 일관성)
        rating_std = np.std([p['scope'] for p in reviewed_products]) if len(reviewed_products) > 1 else 0

        # 평균 상호작용 점수
        avg_interaction = np.mean([p.get('interaction_score', p['scope']) for p in reviewed_products])

        # 고평점 비율 (4점 이상)
        high_rating_ratio = sum(1 for p in reviewed_products if p['scope'] >= 4) / len(reviewed_products)

        behavioral_vector = np.array([
            np.log1p(review_count),  # 로그 스케일 리뷰 수
            avg_rating / 5.0,        # 정규화된 평균 평점
            rating_std / 2.0,        # 정규화된 표준편차
            avg_interaction / 5.0,   # 정규화된 상호작용 점수
            high_rating_ratio        # 고평점 비율
        ])

        return behavioral_vector

    def extract_user_preference_features(
        self,
        user_profile: Dict,
        product_info_dict: Dict
    ) -> np.ndarray:
        """
        사용자 선호도 특성 추출 (구매한 상품들의 특성 집계)

        Args:
            user_profile: 사용자 프로필
            product_info_dict: 전체 상품 정보 딕셔너리

        Returns:
            선호도 특성 벡터
        """
        reviewed_products = user_profile.get('reviewed_products', [])

        if not reviewed_products:
            return np.zeros(len(self.SKIN_TYPES) + len(self.AGE_GROUPS))

        # 구매한 상품들의 대표 피부 타입 집계
        skin_type_scores = np.zeros(len(self.SKIN_TYPES))
        age_group_scores = np.zeros(len(self.AGE_GROUPS))

        for product in reviewed_products:
            product_name = product['product_name']
            if product_name in product_info_dict:
                product_info = product_info_dict[product_name]
                weight = product.get('interaction_score', product['scope']) / 5.0

                # 상품의 대표 피부 타입에 가중치 부여
                for i, st in enumerate(product_info.get('primary_skin_types', [])):
                    if st in self.skin_type_encoder:
                        skin_type_scores[self.skin_type_encoder[st]] += weight * (3 - i) / 3

                # 상품의 대표 연령대에 가중치 부여
                for i, ag in enumerate(product_info.get('primary_age_groups', [])):
                    if ag in self.age_group_encoder:
                        age_group_scores[self.age_group_encoder[ag]] += weight * (3 - i) / 3

        # 정규화
        if skin_type_scores.sum() > 0:
            skin_type_scores = skin_type_scores / skin_type_scores.sum()
        if age_group_scores.sum() > 0:
            age_group_scores = age_group_scores / age_group_scores.sum()

        preference_vector = np.concatenate([skin_type_scores, age_group_scores])
        return preference_vector

    def extract_product_attribute_features(self, product_info: Dict) -> np.ndarray:
        """
        상품 속성 특성 추출

        Args:
            product_info: 상품 정보 딕셔너리

        Returns:
            상품 속성 벡터
        """
        # 리뷰 수 (로그 스케일)
        review_count = np.log1p(product_info.get('review_count', 0))

        # 평균 평점
        avg_rating = product_info.get('avg_rating', 3.0) / 5.0

        # 평균 분석 점수
        max_analytics = 300  # 추정 최대값
        avg_analytics = min(product_info.get('avg_analytics_score', 0) / max_analytics, 1.0)

        # 총 추천 수 (로그 스케일)
        total_recommends = np.log1p(product_info.get('total_recommends', 0))

        # 대표 피부 타입 벡터
        skin_type_vec = np.zeros(len(self.SKIN_TYPES))
        for i, st in enumerate(product_info.get('primary_skin_types', [])):
            if st in self.skin_type_encoder:
                skin_type_vec[self.skin_type_encoder[st]] = (3 - i) / 3

        # 대표 연령대 벡터
        age_group_vec = np.zeros(len(self.AGE_GROUPS))
        for i, ag in enumerate(product_info.get('primary_age_groups', [])):
            if ag in self.age_group_encoder:
                age_group_vec[self.age_group_encoder[ag]] = (3 - i) / 3

        attribute_vector = np.concatenate([
            np.array([review_count, avg_rating, avg_analytics, total_recommends]),
            skin_type_vec,
            age_group_vec
        ])

        return attribute_vector

    def extract_review_text_features(self, texts: List[str]) -> np.ndarray:
        """
        리뷰 텍스트 TF-IDF 특성 추출

        Args:
            texts: 리뷰 텍스트 리스트

        Returns:
            TF-IDF 벡터 행렬
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")

        return self.tfidf_vectorizer.transform(texts).toarray()

    def extract_survey_features(self, survey_summary: Dict) -> np.ndarray:
        """
        설문 응답 특성 추출

        Args:
            survey_summary: 설문 응답 집계 딕셔너리

        Returns:
            설문 특성 벡터
        """
        # 주요 설문 카테고리 및 응답 매핑
        survey_categories = {
            '보습감': ['촉촉해요', '적당해요', '가벼워요'],
            '향': ['향이 좋아요', '향이 적당해요', '향이 없어요'],
            '민감성': ['순해요', '보통이에요', '자극이 있어요'],
            '효과': ['효과가 좋아요', '보통이에요', '효과가 없어요']
        }

        survey_vector = []

        for category, responses in survey_categories.items():
            category_scores = np.zeros(len(responses))

            if category in survey_summary:
                total = sum(survey_summary[category].values())
                if total > 0:
                    for i, response in enumerate(responses):
                        for actual_response, count in survey_summary[category].items():
                            # 유사한 응답 매칭
                            if response in actual_response or actual_response in response:
                                category_scores[i] = count / total
                                break

            survey_vector.extend(category_scores)

        return np.array(survey_vector)

    def extract_all_user_features(
        self,
        user_profile: Dict,
        product_info_dict: Dict
    ) -> np.ndarray:
        """
        사용자의 모든 특성 추출 및 결합

        Args:
            user_profile: 사용자 프로필
            product_info_dict: 상품 정보 딕셔너리

        Returns:
            통합 사용자 특성 벡터
        """
        demographic = self.extract_user_demographic_features(user_profile)
        behavioral = self.extract_user_behavioral_features(user_profile)
        preference = self.extract_user_preference_features(user_profile, product_info_dict)

        return np.concatenate([demographic, behavioral, preference])

    def extract_all_product_features(
        self,
        product_info: Dict,
        review_texts: List[str] = None
    ) -> np.ndarray:
        """
        상품의 모든 특성 추출 및 결합

        Args:
            product_info: 상품 정보
            review_texts: 상품 리뷰 텍스트 리스트

        Returns:
            통합 상품 특성 벡터
        """
        attribute = self.extract_product_attribute_features(product_info)
        survey = self.extract_survey_features(product_info.get('survey_summary', {}))

        # 리뷰 텍스트 특성 (평균)
        if review_texts and self.is_fitted:
            text_features = self.extract_review_text_features(review_texts)
            avg_text_features = np.mean(text_features, axis=0)
        else:
            avg_text_features = np.zeros(self.tfidf_max_features)

        return np.concatenate([attribute, survey, avg_text_features])

    def compute_user_product_similarity(
        self,
        user_features: np.ndarray,
        product_features: np.ndarray,
        weights: Dict[str, float] = None
    ) -> float:
        """
        사용자-상품 유사도 계산

        Args:
            user_features: 사용자 특성 벡터
            product_features: 상품 특성 벡터
            weights: 특성별 가중치

        Returns:
            유사도 점수 (0~1)
        """
        if weights is None:
            weights = {
                'demographic': 0.3,
                'behavioral': 0.2,
                'preference': 0.3,
                'attribute': 0.2
            }

        # 벡터 길이 맞추기 (더 작은 벡터에 패딩)
        max_len = max(len(user_features), len(product_features))
        user_padded = np.pad(user_features, (0, max_len - len(user_features)))
        product_padded = np.pad(product_features, (0, max_len - len(product_features)))

        # 코사인 유사도
        dot_product = np.dot(user_padded, product_padded)
        norm_user = np.linalg.norm(user_padded)
        norm_product = np.linalg.norm(product_padded)

        if norm_user == 0 or norm_product == 0:
            return 0.0

        cosine_sim = dot_product / (norm_user * norm_product)

        # 0~1 범위로 정규화
        return (cosine_sim + 1) / 2

    def build_user_feature_matrix(
        self,
        user_profiles: Dict,
        product_info_dict: Dict
    ) -> Tuple[np.ndarray, List[int]]:
        """
        전체 사용자 특성 행렬 구축

        Args:
            user_profiles: 사용자 프로필 딕셔너리
            product_info_dict: 상품 정보 딕셔너리

        Returns:
            - 사용자 특성 행렬
            - 사용자 ID 리스트
        """
        user_ids = list(user_profiles.keys())
        feature_list = []

        for user_id in user_ids:
            features = self.extract_all_user_features(
                user_profiles[user_id],
                product_info_dict
            )
            feature_list.append(features)

        return np.array(feature_list), user_ids

    def build_product_feature_matrix(
        self,
        product_info_dict: Dict,
        processed_df: pd.DataFrame = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        전체 상품 특성 행렬 구축

        Args:
            product_info_dict: 상품 정보 딕셔너리
            processed_df: 전처리된 DataFrame (리뷰 텍스트용)

        Returns:
            - 상품 특성 행렬
            - 상품명 리스트
        """
        product_names = list(product_info_dict.keys())
        feature_list = []

        for product_name in product_names:
            # 해당 상품의 리뷰 텍스트
            if processed_df is not None:
                product_reviews = processed_df[
                    processed_df['prodName'] == product_name
                ]['cleaned_review'].tolist()
            else:
                product_reviews = None

            features = self.extract_all_product_features(
                product_info_dict[product_name],
                product_reviews
            )
            feature_list.append(features)

        return np.array(feature_list), product_names


if __name__ == "__main__":
    # 테스트
    from data_processor import DataProcessor, create_sample_dataframe

    # 데이터 로드 및 처리
    processor = DataProcessor()
    df = create_sample_dataframe()
    processed_df = processor.process_data(df)

    # 특성 추출기 초기화 및 학습
    extractor = FeatureExtractor(tfidf_max_features=50)
    extractor.fit(processed_df)

    # 사용자 특성 추출 테스트
    for member_sn, profile in processor.user_profiles.items():
        user_features = extractor.extract_all_user_features(
            profile,
            processor.product_info
        )
        print(f"User {member_sn} features shape: {user_features.shape}")
        break

    # 상품 특성 추출 테스트
    for product_name, info in processor.product_info.items():
        product_reviews = processed_df[
            processed_df['prodName'] == product_name
        ]['cleaned_review'].tolist()

        product_features = extractor.extract_all_product_features(
            info,
            product_reviews
        )
        print(f"Product '{product_name[:30]}...' features shape: {product_features.shape}")
        break

    # 특성 행렬 구축
    user_matrix, user_ids = extractor.build_user_feature_matrix(
        processor.user_profiles,
        processor.product_info
    )
    print(f"\nUser feature matrix shape: {user_matrix.shape}")

    product_matrix, product_names = extractor.build_product_feature_matrix(
        processor.product_info,
        processed_df
    )
    print(f"Product feature matrix shape: {product_matrix.shape}")
