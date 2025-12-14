"""
데이터 전처리 모듈
- 리뷰 데이터 파싱 및 정제
- 사용자 속성 정보 추출
- 설문 응답 파싱
"""

import pandas as pd
import numpy as np
import re
import ast
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class UserProfile:
    """사용자 프로필 데이터 클래스"""
    member_sn: int
    age_group: str
    gender: str
    skin_type: str
    skin_concerns: List[str]

    def to_dict(self) -> Dict:
        return {
            'member_sn': self.member_sn,
            'age_group': self.age_group,
            'gender': self.gender,
            'skin_type': self.skin_type,
            'skin_concerns': self.skin_concerns
        }


@dataclass
class ReviewData:
    """리뷰 데이터 클래스"""
    review_sn: int
    product_name: str
    member_sn: int
    scope: int
    review_text: str
    survey_responses: Dict[str, str]
    analytics_score: float
    recommend_cnt: int

    def to_dict(self) -> Dict:
        return {
            'review_sn': self.review_sn,
            'product_name': self.product_name,
            'member_sn': self.member_sn,
            'scope': self.scope,
            'review_text': self.review_text,
            'survey_responses': self.survey_responses,
            'analytics_score': self.analytics_score,
            'recommend_cnt': self.recommend_cnt
        }


class DataProcessor:
    """
    리뷰 데이터 전처리 클래스

    주요 기능:
    - 원본 데이터 로드 및 정제
    - 사용자 속성 정보 파싱
    - 설문 응답 파싱
    - 사용자-상품 상호작용 매트릭스 생성
    """

    # 피부 타입 매핑
    SKIN_TYPE_MAP = {
        '건성': 'dry',
        '지성': 'oily',
        '복합성': 'combination',
        '중성': 'normal',
        '민감성': 'sensitive'
    }

    # 피부 고민 매핑
    SKIN_CONCERN_MAP = {
        '주름': 'wrinkle',
        '탄력': 'elasticity',
        '미백': 'whitening',
        '모공': 'pore',
        '트러블': 'trouble',
        '각질': 'dead_skin',
        '건조': 'dryness',
        '유분': 'oiliness',
        '민감': 'sensitivity',
        '잡티': 'blemish',
        '다크서클': 'dark_circle'
    }

    # 연령대 매핑
    AGE_GROUP_MAP = {
        '10대': '10s',
        '20대': '20s',
        '30대': '30s',
        '40대': '40s',
        '50대': '50s',
        '50대 이상': '50+',
        '60대': '60s',
        '60대 이상': '60+'
    }

    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.user_profiles = {}
        self.product_info = {}

    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        데이터 파일 로드

        Args:
            file_path: 데이터 파일 경로
            file_type: 파일 유형 ('csv', 'excel', 'json')

        Returns:
            로드된 DataFrame
        """
        if file_type == 'csv':
            self.raw_data = pd.read_csv(file_path, encoding='utf-8')
        elif file_type == 'excel':
            self.raw_data = pd.read_excel(file_path)
        elif file_type == 'json':
            self.raw_data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        print(f"Loaded {len(self.raw_data)} records")
        return self.raw_data

    def parse_user_attributes(self, attr_string: str) -> Dict[str, Any]:
        """
        사용자 속성 문자열 파싱

        예시 입력: "50대 이상/여성/건성/주름"

        Args:
            attr_string: 사용자 속성 문자열

        Returns:
            파싱된 속성 딕셔너리
        """
        result = {
            'age_group': None,
            'gender': None,
            'skin_type': None,
            'skin_concerns': []
        }

        if pd.isna(attr_string) or not attr_string:
            return result

        parts = str(attr_string).split('/')

        for part in parts:
            part = part.strip()

            # 연령대 확인
            for age_kr, age_en in self.AGE_GROUP_MAP.items():
                if age_kr in part:
                    result['age_group'] = age_en
                    break

            # 성별 확인
            if '여성' in part:
                result['gender'] = 'female'
            elif '남성' in part:
                result['gender'] = 'male'

            # 피부 타입 확인
            for skin_kr, skin_en in self.SKIN_TYPE_MAP.items():
                if skin_kr == part:
                    result['skin_type'] = skin_en
                    break

            # 피부 고민 확인
            for concern_kr, concern_en in self.SKIN_CONCERN_MAP.items():
                if concern_kr == part:
                    result['skin_concerns'].append(concern_en)

        return result

    def parse_surveys(self, survey_data: Any) -> Dict[str, str]:
        """
        설문 응답 데이터 파싱

        Args:
            survey_data: 설문 응답 리스트 (문자열 또는 리스트)

        Returns:
            설문 응답 딕셔너리
        """
        result = {}

        if pd.isna(survey_data) or not survey_data:
            return result

        # 문자열인 경우 파싱
        if isinstance(survey_data, str):
            try:
                survey_data = ast.literal_eval(survey_data)
            except (ValueError, SyntaxError):
                return result

        if not isinstance(survey_data, list):
            return result

        for item in survey_data:
            if isinstance(item, dict):
                question = item.get('questionHeader', '')
                answer = item.get('responseBodyText', '')
                if question and answer:
                    # 피부타입/고민 관련 질문은 별도 처리
                    if '피부타입' in question or '피부고민' in question:
                        continue
                    result[question] = answer

        return result

    def process_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        전체 데이터 전처리 수행

        Args:
            df: 처리할 DataFrame (None이면 self.raw_data 사용)

        Returns:
            처리된 DataFrame
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data to process. Load data first.")

        processed_df = df.copy()

        # 결측치 처리
        processed_df['prodReviewBodyText'] = processed_df['prodReviewBodyText'].fillna('')
        processed_df['scope'] = processed_df['scope'].fillna(3)
        processed_df['recommendCnt'] = processed_df['recommendCnt'].fillna(0)
        processed_df['rvAnalyticsScore'] = processed_df['rvAnalyticsScore'].fillna(0)

        # 사용자 속성 파싱
        user_attrs = processed_df['userAddAttrInfo'].apply(self.parse_user_attributes)
        processed_df['age_group'] = user_attrs.apply(lambda x: x['age_group'])
        processed_df['gender'] = user_attrs.apply(lambda x: x['gender'])
        processed_df['skin_type'] = user_attrs.apply(lambda x: x['skin_type'])
        processed_df['skin_concerns'] = user_attrs.apply(lambda x: x['skin_concerns'])

        # 설문 응답 파싱
        processed_df['parsed_surveys'] = processed_df['surveys'].apply(self.parse_surveys)

        # 리뷰 텍스트 정제
        processed_df['cleaned_review'] = processed_df['prodReviewBodyText'].apply(self._clean_text)

        # 상호작용 점수 계산 (평점 + 정규화된 추천수 + 분석점수)
        max_recommend = processed_df['recommendCnt'].max()
        if max_recommend > 0:
            processed_df['normalized_recommend'] = processed_df['recommendCnt'] / max_recommend
        else:
            processed_df['normalized_recommend'] = 0

        max_analytics = processed_df['rvAnalyticsScore'].max()
        if max_analytics > 0:
            processed_df['normalized_analytics'] = processed_df['rvAnalyticsScore'] / max_analytics
        else:
            processed_df['normalized_analytics'] = 0

        # 종합 상호작용 점수 (가중 평균)
        processed_df['interaction_score'] = (
            processed_df['scope'] * 0.6 +
            processed_df['normalized_recommend'] * 5 * 0.2 +
            processed_df['normalized_analytics'] * 5 * 0.2
        )

        self.processed_data = processed_df
        self._build_user_profiles(processed_df)
        self._build_product_info(processed_df)

        print(f"Processed {len(processed_df)} records")
        return processed_df

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not isinstance(text, str):
            return ''

        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 특수문자 제거 (한글, 영문, 숫자, 공백 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _build_user_profiles(self, df: pd.DataFrame):
        """사용자 프로필 구축"""
        for _, row in df.iterrows():
            member_sn = row['memberSn']
            if member_sn not in self.user_profiles:
                self.user_profiles[member_sn] = {
                    'member_sn': member_sn,
                    'age_group': row.get('age_group'),
                    'gender': row.get('gender'),
                    'skin_type': row.get('skin_type'),
                    'skin_concerns': row.get('skin_concerns', []),
                    'reviewed_products': [],
                    'avg_rating': 0
                }

            self.user_profiles[member_sn]['reviewed_products'].append({
                'product_name': row['prodName'],
                'scope': row['scope'],
                'interaction_score': row.get('interaction_score', row['scope'])
            })

        # 평균 평점 계산
        for member_sn, profile in self.user_profiles.items():
            if profile['reviewed_products']:
                avg = np.mean([p['scope'] for p in profile['reviewed_products']])
                profile['avg_rating'] = avg

    def _build_product_info(self, df: pd.DataFrame):
        """상품 정보 구축"""
        for product_name in df['prodName'].unique():
            product_df = df[df['prodName'] == product_name]

            self.product_info[product_name] = {
                'product_name': product_name,
                'review_count': len(product_df),
                'avg_rating': product_df['scope'].mean(),
                'avg_analytics_score': product_df['rvAnalyticsScore'].mean(),
                'total_recommends': product_df['recommendCnt'].sum(),
                # 대표 피부 타입 (가장 많이 구매한 피부 타입)
                'primary_skin_types': product_df['skin_type'].value_counts().head(3).index.tolist(),
                # 대표 연령대
                'primary_age_groups': product_df['age_group'].value_counts().head(3).index.tolist(),
                # 설문 응답 집계
                'survey_summary': self._aggregate_surveys(product_df)
            }

    def _aggregate_surveys(self, product_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """상품별 설문 응답 집계"""
        survey_summary = {}

        for surveys in product_df['parsed_surveys']:
            if isinstance(surveys, dict):
                for question, answer in surveys.items():
                    if question not in survey_summary:
                        survey_summary[question] = {}
                    if answer not in survey_summary[question]:
                        survey_summary[question][answer] = 0
                    survey_summary[question][answer] += 1

        return survey_summary

    def create_user_item_matrix(self) -> Tuple[pd.DataFrame, Dict[int, int], Dict[str, int]]:
        """
        사용자-상품 상호작용 매트릭스 생성

        Returns:
            - 상호작용 매트릭스
            - 사용자 ID 매핑
            - 상품 ID 매핑
        """
        if self.processed_data is None:
            raise ValueError("Process data first")

        # 사용자, 상품 ID 매핑
        unique_users = self.processed_data['memberSn'].unique()
        unique_products = self.processed_data['prodName'].unique()

        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        product_to_idx = {product: idx for idx, product in enumerate(unique_products)}

        # 매트릭스 생성
        matrix = np.zeros((len(unique_users), len(unique_products)))

        for _, row in self.processed_data.iterrows():
            user_idx = user_to_idx[row['memberSn']]
            product_idx = product_to_idx[row['prodName']]
            matrix[user_idx, product_idx] = row['interaction_score']

        matrix_df = pd.DataFrame(
            matrix,
            index=unique_users,
            columns=unique_products
        )

        return matrix_df, user_to_idx, product_to_idx

    def get_user_profile(self, member_sn: int) -> Optional[Dict]:
        """특정 사용자 프로필 조회"""
        return self.user_profiles.get(member_sn)

    def get_product_info(self, product_name: str) -> Optional[Dict]:
        """특정 상품 정보 조회"""
        return self.product_info.get(product_name)

    def get_similar_users_by_attributes(self, member_sn: int, top_n: int = 10) -> List[int]:
        """
        속성 기반 유사 사용자 검색

        Args:
            member_sn: 대상 사용자 ID
            top_n: 반환할 유사 사용자 수

        Returns:
            유사 사용자 ID 리스트
        """
        target_profile = self.user_profiles.get(member_sn)
        if not target_profile:
            return []

        similarity_scores = []

        for other_sn, other_profile in self.user_profiles.items():
            if other_sn == member_sn:
                continue

            score = 0

            # 연령대 일치
            if target_profile['age_group'] == other_profile['age_group']:
                score += 2

            # 성별 일치
            if target_profile['gender'] == other_profile['gender']:
                score += 1

            # 피부 타입 일치
            if target_profile['skin_type'] == other_profile['skin_type']:
                score += 3

            # 피부 고민 일치 (교집합)
            target_concerns = set(target_profile.get('skin_concerns', []))
            other_concerns = set(other_profile.get('skin_concerns', []))
            common_concerns = len(target_concerns & other_concerns)
            score += common_concerns * 2

            similarity_scores.append((other_sn, score))

        # 점수 기준 정렬
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        return [user_sn for user_sn, _ in similarity_scores[:top_n]]


def create_sample_dataframe() -> pd.DataFrame:
    """샘플 데이터프레임 생성"""
    sample_data = {
        'prodReviewSn': [5878822, 5878823, 5878824, 5878825, 5878826],
        'prodReviewTypeCode': ['Pur', 'Pur', 'Pur', 'Pur', 'Pur'],
        'prodReviewTitle': ['', '', '', '', ''],
        'prodReviewBodyText': [
            '건성이고 주름이 많은데 촉촉하게 효과있습니다',
            '지성피부인데 가볍게 발리고 좋아요',
            '민감한 피부에도 자극없이 사용가능해요',
            '보습력이 정말 좋고 향도 은은해요',
            '피부가 환해지는 느낌이에요'
        ],
        'prodName': [
            '[홀리데이] 자음생크림 리치 50ml 단품기획세트',
            '[홀리데이] 자음생크림 라이트 50ml',
            '설화수 윤조에센스',
            '[홀리데이] 자음생크림 리치 50ml 단품기획세트',
            '설화수 자정미라클에센스'
        ],
        'memberSn': [8208875, 8208876, 8208877, 8208875, 8208878],
        'memberStatus': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal'],
        'memberId': ['k200*****', 'j100*****', 'm300*****', 'k200*****', 'l400*****'],
        'scope': [5, 4, 5, 4, 5],
        'recommendCnt': [0, 5, 10, 3, 8],
        'reportCnt': [0, 0, 0, 0, 0],
        'rvAnalyticsScore': [198, 150, 220, 180, 200],
        'recommendYn': ['N', 'N', 'N', 'N', 'N'],
        'reportYn': ['N', 'N', 'N', 'N', 'N'],
        'giftServiceLimitedYn': ['N', 'N', 'N', 'N', 'N'],
        'prodReviewRegistDt': [
            '2025-11-18T14:43:55.2340+0900',
            '2025-11-17T10:20:30.0000+0900',
            '2025-11-16T09:15:20.0000+0900',
            '2025-11-15T16:30:45.0000+0900',
            '2025-11-14T11:45:10.0000+0900'
        ],
        'oneLineDesc': ['', '', '', '', ''],
        'tipDoc': ['', '', '', '', ''],
        'userAddAttrInfo': [
            '50대 이상/여성/건성/주름',
            '20대/여성/지성/모공',
            '30대/여성/민감성/트러블',
            '50대 이상/여성/건성/주름',
            '40대/여성/복합성/미백'
        ],
        'naverId': ['', '', '', '', ''],
        'imgList': ['[]', '[]', '[]', '[]', '[]'],
        'surveys': [
            "[{'questionHeader': '보습감', 'responseBodyText': '촉촉해요', 'memberAttrTgtYn': 'N'}]",
            "[{'questionHeader': '보습감', 'responseBodyText': '가벼워요', 'memberAttrTgtYn': 'N'}]",
            "[{'questionHeader': '민감성', 'responseBodyText': '순해요', 'memberAttrTgtYn': 'N'}]",
            "[{'questionHeader': '향', 'responseBodyText': '향이 좋아요', 'memberAttrTgtYn': 'N'}]",
            "[{'questionHeader': '효과', 'responseBodyText': '피부톤이 밝아져요', 'memberAttrTgtYn': 'N'}]"
        ],
        'isBlock': [False, False, False, False, False],
        'profile.nickName': ['k200*****', 'j100*****', 'm300*****', 'k200*****', 'l400*****'],
        'profile.imageUrl': ['', '', '', '', ''],
        'profile.gradeName': ['BEGINNER', 'BEGINNER', 'MEMBER', 'BEGINNER', 'MEMBER'],
        'profile.badgeName': ['', '', '', '', '']
    }

    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # 테스트
    processor = DataProcessor()
    df = create_sample_dataframe()

    # 데이터 처리
    processed = processor.process_data(df)
    print("\n처리된 데이터 컬럼:", processed.columns.tolist())

    # 사용자 프로필 확인
    print("\n사용자 프로필:")
    for member_sn, profile in processor.user_profiles.items():
        print(f"  {member_sn}: {profile['skin_type']}, {profile['age_group']}")

    # 상품 정보 확인
    print("\n상품 정보:")
    for product_name, info in processor.product_info.items():
        print(f"  {product_name[:30]}...: 평점 {info['avg_rating']:.2f}")

    # 상호작용 매트릭스
    matrix, user_map, product_map = processor.create_user_item_matrix()
    print(f"\n상호작용 매트릭스 크기: {matrix.shape}")
