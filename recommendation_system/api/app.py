"""
FastAPI 기반 추천 시스템 REST API
================================

엔드포인트:
- POST /api/v1/recommendations: 개인화 추천
- POST /api/v1/similar: 유사 상품 추천
- GET /api/v1/trending: 인기 상품
- POST /api/v1/explain: 추천 설명
- GET /api/v1/user/{user_id}: 사용자 프로필
- GET /api/v1/product/{product_name}: 상품 정보
- GET /api/v1/health: 헬스 체크

실행:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.data_processor import DataProcessor
from utils.feature_extractor import FeatureExtractor
from models.hybrid_recommender import HybridRecommender


# ============================================================
# Pydantic 모델 정의
# ============================================================

class RecommendationRequest(BaseModel):
    """추천 요청 모델"""
    user_id: int = Field(..., description="사용자 ID")
    top_n: int = Field(default=10, ge=1, le=50, description="추천 상품 수")
    exclude_purchased: bool = Field(default=True, description="구매한 상품 제외 여부")


class SimilarProductRequest(BaseModel):
    """유사 상품 요청 모델"""
    product_name: str = Field(..., description="기준 상품명")
    user_id: Optional[int] = Field(default=None, description="사용자 ID (개인화용)")
    top_n: int = Field(default=5, ge=1, le=20, description="추천 상품 수")


class ExplainRequest(BaseModel):
    """추천 설명 요청 모델"""
    user_id: int = Field(..., description="사용자 ID")
    product_name: str = Field(..., description="상품명")


class RecommendationResponse(BaseModel):
    """추천 응답 모델"""
    product_name: str
    final_score: float
    cf_score: float
    cbf_score: float
    mf_score: float
    explanation: str
    confidence: float
    recommendation_type: str


class UserProfileResponse(BaseModel):
    """사용자 프로필 응답 모델"""
    member_sn: int
    skin_type: Optional[str]
    age_group: Optional[str]
    gender: Optional[str]
    skin_concerns: List[str]
    avg_rating: float
    review_count: int


class ProductInfoResponse(BaseModel):
    """상품 정보 응답 모델"""
    product_name: str
    review_count: int
    avg_rating: float
    total_recommends: int
    primary_skin_types: List[str]
    primary_age_groups: List[str]


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    model_loaded: bool
    n_users: int
    n_products: int


# ============================================================
# FastAPI 앱 초기화
# ============================================================

app = FastAPI(
    title="개인화 상품 추천 API",
    description="""
    사용자 정보 기반 하이브리드 상품 추천 시스템 API

    ## 주요 기능
    - **개인화 추천**: 사용자 프로필 기반 맞춤 추천
    - **유사 상품**: 상품 간 유사도 기반 추천
    - **인기 상품**: 트렌딩/인기 상품 추천
    - **추천 설명**: 왜 이 상품을 추천했는지 설명

    ## 추천 알고리즘
    - 협업 필터링 (Collaborative Filtering)
    - 콘텐츠 기반 필터링 (Content-based Filtering)
    - 행렬 분해 (Matrix Factorization)
    - 하이브리드 앙상블
    """,
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 전역 변수 및 모델 초기화
# ============================================================

recommender: Optional[HybridRecommender] = None
processor: Optional[DataProcessor] = None


def initialize_model():
    """모델 초기화"""
    global recommender, processor

    print("모델 초기화 중...")

    processor = DataProcessor()

    # 샘플 데이터 로드
    sample_file = os.path.join(project_root, 'data', 'sample_reviews.csv')

    if os.path.exists(sample_file):
        df = processor.load_data(sample_file)
    else:
        from utils.data_processor import create_sample_dataframe
        df = create_sample_dataframe()

    processed_df = processor.process_data(df)
    matrix, _, _ = processor.create_user_item_matrix()

    # 특성 추출기
    extractor = FeatureExtractor(tfidf_max_features=50)
    extractor.fit(processed_df)

    # 추천 시스템
    recommender = HybridRecommender(
        cf_weight=0.35,
        cbf_weight=0.35,
        mf_weight=0.30
    )
    recommender.fit(
        processor.user_profiles,
        processor.product_info,
        matrix,
        processed_df,
        extractor,
        verbose=False
    )

    print("모델 초기화 완료!")


@app.on_event("startup")
async def startup_event():
    """앱 시작 시 모델 초기화"""
    initialize_model()


# ============================================================
# API 엔드포인트
# ============================================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["시스템"])
async def health_check():
    """
    헬스 체크

    서버 상태 및 모델 로드 여부 확인
    """
    return HealthResponse(
        status="healthy" if recommender else "unhealthy",
        model_loaded=recommender is not None,
        n_users=len(processor.user_profiles) if processor else 0,
        n_products=len(processor.product_info) if processor else 0
    )


@app.post("/api/v1/recommendations", response_model=List[RecommendationResponse], tags=["추천"])
async def get_recommendations(request: RecommendationRequest):
    """
    개인화된 상품 추천

    사용자의 프로필(피부 타입, 연령대, 피부 고민)과 구매 이력을 기반으로
    하이브리드 알고리즘을 통해 맞춤형 상품을 추천합니다.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    recommendations = recommender.recommend(
        user_id=request.user_id,
        top_n=request.top_n,
        exclude_purchased=request.exclude_purchased
    )

    return [
        RecommendationResponse(
            product_name=rec.product_name,
            final_score=round(rec.final_score, 2),
            cf_score=round(rec.cf_score, 2),
            cbf_score=round(rec.cbf_score, 2),
            mf_score=round(rec.mf_score, 2),
            explanation=rec.explanation,
            confidence=round(rec.confidence, 2),
            recommendation_type=rec.recommendation_type.value
        )
        for rec in recommendations
    ]


@app.post("/api/v1/similar", response_model=List[RecommendationResponse], tags=["추천"])
async def get_similar_products(request: SimilarProductRequest):
    """
    유사 상품 추천

    특정 상품과 유사한 상품을 추천합니다.
    사용자 ID가 제공되면 개인화 점수도 함께 반영합니다.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    if request.product_name not in processor.product_info:
        raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다")

    recommendations = recommender.recommend_similar(
        product_name=request.product_name,
        user_id=request.user_id,
        top_n=request.top_n
    )

    return [
        RecommendationResponse(
            product_name=rec.product_name,
            final_score=round(rec.final_score, 4),
            cf_score=round(rec.cf_score, 4),
            cbf_score=round(rec.cbf_score, 4),
            mf_score=round(rec.mf_score, 4),
            explanation=rec.explanation,
            confidence=round(rec.confidence, 2),
            recommendation_type=rec.recommendation_type.value
        )
        for rec in recommendations
    ]


@app.get("/api/v1/trending", response_model=List[RecommendationResponse], tags=["추천"])
async def get_trending_products(top_n: int = Query(default=10, ge=1, le=50)):
    """
    인기/트렌딩 상품

    리뷰 수, 평점, 추천 수를 기반으로 인기 상품을 반환합니다.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    recommendations = recommender.recommend_trending(top_n=top_n)

    return [
        RecommendationResponse(
            product_name=rec.product_name,
            final_score=round(rec.final_score, 2),
            cf_score=0,
            cbf_score=0,
            mf_score=0,
            explanation=rec.explanation,
            confidence=round(rec.confidence, 2),
            recommendation_type=rec.recommendation_type.value
        )
        for rec in recommendations
    ]


@app.post("/api/v1/explain", tags=["추천"])
async def explain_recommendation(request: ExplainRequest) -> Dict[str, Any]:
    """
    추천 설명

    특정 사용자에게 특정 상품이 왜 추천되었는지 상세하게 설명합니다.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    if request.product_name not in processor.product_info:
        raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다")

    explanation = recommender.explain_recommendation(
        user_id=request.user_id,
        product_name=request.product_name
    )

    return explanation


@app.get("/api/v1/user/{user_id}", response_model=UserProfileResponse, tags=["데이터"])
async def get_user_profile(user_id: int):
    """
    사용자 프로필 조회

    사용자의 프로필 정보(피부 타입, 연령대, 피부 고민 등)를 조회합니다.
    """
    if not processor:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    profile = processor.user_profiles.get(user_id)

    if not profile:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

    return UserProfileResponse(
        member_sn=profile['member_sn'],
        skin_type=profile.get('skin_type'),
        age_group=profile.get('age_group'),
        gender=profile.get('gender'),
        skin_concerns=profile.get('skin_concerns', []),
        avg_rating=round(profile.get('avg_rating', 0), 2),
        review_count=len(profile.get('reviewed_products', []))
    )


@app.get("/api/v1/product/{product_name}", response_model=ProductInfoResponse, tags=["데이터"])
async def get_product_info(product_name: str):
    """
    상품 정보 조회

    상품의 상세 정보(평점, 리뷰 수, 대표 피부 타입 등)를 조회합니다.
    """
    if not processor:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    info = processor.product_info.get(product_name)

    if not info:
        raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다")

    return ProductInfoResponse(
        product_name=info['product_name'],
        review_count=info.get('review_count', 0),
        avg_rating=round(info.get('avg_rating', 0), 2),
        total_recommends=info.get('total_recommends', 0),
        primary_skin_types=info.get('primary_skin_types', []),
        primary_age_groups=info.get('primary_age_groups', [])
    )


@app.get("/api/v1/users", tags=["데이터"])
async def list_users(limit: int = Query(default=10, ge=1, le=100)):
    """
    사용자 목록 조회

    시스템에 등록된 사용자 목록을 조회합니다.
    """
    if not processor:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    users = list(processor.user_profiles.keys())[:limit]

    return {
        "total": len(processor.user_profiles),
        "users": [
            {
                "user_id": uid,
                "skin_type": processor.user_profiles[uid].get('skin_type'),
                "age_group": processor.user_profiles[uid].get('age_group')
            }
            for uid in users
        ]
    }


@app.get("/api/v1/products", tags=["데이터"])
async def list_products(limit: int = Query(default=10, ge=1, le=100)):
    """
    상품 목록 조회

    시스템에 등록된 상품 목록을 조회합니다.
    """
    if not processor:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    products = list(processor.product_info.keys())[:limit]

    return {
        "total": len(processor.product_info),
        "products": [
            {
                "product_name": pname,
                "avg_rating": round(processor.product_info[pname].get('avg_rating', 0), 2),
                "review_count": processor.product_info[pname].get('review_count', 0)
            }
            for pname in products
        ]
    }


@app.get("/api/v1/model/info", tags=["시스템"])
async def get_model_info():
    """
    모델 정보 조회

    현재 로드된 추천 모델의 설정 및 성능 정보를 조회합니다.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    return recommender.get_model_performance()


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
