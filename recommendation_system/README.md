# 개인화된 하이브리드 추천 시스템

사용자 정보를 입력으로 받아 개인화된 상품을 추천하는 하이브리드 추천 시스템입니다.

## 🎯 주요 특징

- **개인화된 추천**: 각 사용자의 고유한 선호도와 행동 패턴을 분석
- **하이브리드 기법**: 협업 필터링과 컨텐츠 기반 필터링을 결합하여 최적의 추천 제공
- **다양한 전략**: 가중 평균, 스위칭, 혼합 등 상황에 맞는 추천 전략 선택 가능
- **확장 가능**: 모듈화된 설계로 새로운 알고리즘 추가 용이

## 📁 프로젝트 구조

```
recommendation_system/
├── data/                           # 데이터 모델 및 샘플 데이터
│   ├── data_models.py             # User, Product, Interaction 데이터 클래스
│   └── sample_data.py             # 샘플 데이터 생성기
├── models/                         # 추천 알고리즘 모델
│   ├── collaborative_filtering.py # 협업 필터링 (User/Item-based, MF)
│   ├── content_based_filtering.py # 컨텐츠 기반 필터링 (TF-IDF, Feature)
│   └── hybrid_recommender.py      # 하이브리드 추천 엔진
├── docs/                           # 문서
│   ├── FLOWCHART.md               # 시스템 흐름도
│   └── METHODOLOGY.md             # 방법론 설명 및 성능 비교
├── main.py                         # 메인 실행 파일
├── requirements.txt                # 필요 패키지
└── README.md                       # 프로젝트 설명 (현재 문서)
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# Python 3.8 이상 필요
python --version

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 실행

```bash
# 메인 추천 시스템 실행
cd recommendation_system
python main.py
```

### 3. 사용 예시

프로그램을 실행하면:
1. 자동으로 샘플 데이터셋 생성 (사용자 100명, 상품 200개)
2. 추천 엔진 초기화
3. 다양한 사용자 프로필에 대한 추천 결과 출력
4. 인터랙티브 모드에서 원하는 사용자 ID 입력하여 추천받기

## 🧩 구현된 알고리즘

### 협업 필터링 (Collaborative Filtering)

#### 1. User-based CF
- 유사한 사용자들의 선호도를 기반으로 추천
- 코사인 유사도 및 피어슨 상관계수 지원

#### 2. Item-based CF
- 유사한 상품들의 패턴을 기반으로 추천
- 안정적이고 확장 가능

#### 3. Matrix Factorization (SVD)
- 잠재 요인 분해를 통한 고급 협업 필터링
- 높은 정확도와 희소성 문제 완화

### 컨텐츠 기반 필터링 (Content-based Filtering)

#### 1. TF-IDF 기반 필터
- 텍스트 정보(이름, 설명, 태그) 분석
- 상품 간 유사도 계산

#### 2. Feature 기반 필터
- 구조화된 특성(카테고리, 가격, 브랜드) 활용
- 사용자 프로필과 상품 특성 매칭

### 하이브리드 전략

#### 1. 가중 평균 (Weighted Hybrid)
- 모든 알고리즘의 결과를 가중치로 결합
- 정확도 최우선

#### 2. 스위칭 (Switching Hybrid)
- 사용자 상태에 따라 최적의 알고리즘 선택
- Cold Start 대응 및 효율성

#### 3. 혼합 (Mixed Hybrid)
- 여러 알고리즘에서 일부씩 추천
- 다양성 극대화

## 📊 성능 비교

| 알고리즘 | Cold Start | 정확도 | 다양성 | 속도 |
|----------|------------|--------|--------|------|
| User-based CF | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Item-based CF | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Matrix Factorization | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Content-based | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Weighted Hybrid** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

자세한 성능 비교는 [METHODOLOGY.md](docs/METHODOLOGY.md)를 참고하세요.

## 🎓 왜 하이브리드 방식인가?

### 단일 알고리즘의 한계

- **협업 필터링**: Cold Start 문제, 데이터 희소성
- **컨텐츠 기반**: Over-specialization, 다양성 부족

### 하이브리드의 장점

1. **상호 보완**: 각 알고리즘의 약점을 다른 알고리즘으로 보완
2. **안정성**: 일관되고 신뢰할 수 있는 추천
3. **유연성**: 상황에 따라 전략 조정 가능
4. **검증된 방식**: Netflix, Amazon 등 대형 플랫폼에서 사용

## 📈 실전 활용 예시

### E-commerce
```python
# Item-based CF + Content-based
weighted_recommender = WeightedHybridRecommender(
    dataset,
    weights={'item_cf': 0.6, 'content': 0.4}
)
```

### 미디어 스트리밍
```python
# MF + User-based CF
weighted_recommender = WeightedHybridRecommender(
    dataset,
    weights={'mf': 0.5, 'user_cf': 0.5}
)
```

### 뉴스/기사
```python
# Switching (Cold Start 대응)
switching_recommender = SwitchingHybridRecommender(dataset)
```

## 📚 문서

- [시스템 흐름도](docs/FLOWCHART.md) - Mermaid 다이어그램으로 작성된 전체 시스템 흐름
- [방법론 설명](docs/METHODOLOGY.md) - 알고리즘 상세 설명 및 성능 비교

## 🔧 커스터마이징

### 가중치 조정

```python
# 가중치 변경 예시
custom_weights = {
    'user_cf': 0.15,
    'item_cf': 0.25,
    'mf': 0.35,
    'content': 0.25
}

recommender = WeightedHybridRecommender(dataset, weights=custom_weights)
```

### 새로운 알고리즘 추가

1. `models/` 디렉토리에 새 모듈 추가
2. `recommend()` 메서드 구현
3. `hybrid_recommender.py`에서 통합

## 🧪 테스트

### 샘플 데이터 생성 테스트
```bash
python data/sample_data.py
```

### 개별 모듈 테스트
```python
from data.sample_data import create_sample_dataset
from models.collaborative_filtering import MatrixFactorization

dataset = create_sample_dataset()
mf = MatrixFactorization(dataset.rating_matrix, n_factors=20)
recommendations = mf.recommend(user_idx=0, n_recommendations=10)
```

## 📝 라이센스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 👥 기여

개선 사항이나 버그 리포트는 언제든 환영합니다!

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**Version**: 1.0
**Last Updated**: 2025-12-14
**Author**: Vibe Coding Team
