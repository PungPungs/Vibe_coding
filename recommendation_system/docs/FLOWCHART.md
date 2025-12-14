# 하이브리드 추천 시스템 흐름도

## 전체 시스템 아키텍처

```mermaid
graph TB
    Start([시스템 시작]) --> LoadData[데이터 로드]
    LoadData --> InitModels[추천 모델 초기화]

    InitModels --> UserCF[User-based CF]
    InitModels --> ItemCF[Item-based CF]
    InitModels --> MF[Matrix Factorization]
    InitModels --> Content[Content-based Filter]

    UserCF --> Hybrid{하이브리드<br/>전략 선택}
    ItemCF --> Hybrid
    MF --> Hybrid
    Content --> Hybrid

    Hybrid -->|가중 평균| Weighted[WeightedHybrid]
    Hybrid -->|스위칭| Switching[SwitchingHybrid]
    Hybrid -->|혼합| Mixed[MixedHybrid]

    Weighted --> Result[추천 결과]
    Switching --> Result
    Mixed --> Result

    Result --> Display[결과 출력]
    Display --> End([종료])

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style Hybrid fill:#fff4e1
    style Result fill:#e1f0ff
```

## 데이터 처리 흐름

```mermaid
graph LR
    subgraph "데이터 수집"
        A[사용자 정보] --> D[Dataset]
        B[상품 정보] --> D
        C[상호작용 데이터] --> D
    end

    subgraph "데이터 전처리"
        D --> E[User-Item<br/>Rating Matrix]
        E --> F[정규화]
        F --> G[희소성 처리]
    end

    subgraph "모델 입력"
        G --> H[협업 필터링]
        G --> I[컨텐츠 필터링]
    end

    style D fill:#e1f5e1
    style E fill:#ffe1e1
    style G fill:#e1f0ff
```

## 협업 필터링 처리 흐름

```mermaid
graph TB
    subgraph "User-based CF"
        U1[사용자 평점 벡터] --> U2[유사도 계산<br/>코사인/피어슨]
        U2 --> U3[k-최근접 이웃 선택]
        U3 --> U4[가중 평균<br/>예측 평점 계산]
    end

    subgraph "Item-based CF"
        I1[상품 평점 벡터] --> I2[아이템 간<br/>유사도 계산]
        I2 --> I3[유사 상품 선택]
        I3 --> I4[가중 평균<br/>예측 평점 계산]
    end

    subgraph "Matrix Factorization"
        M1[Rating Matrix] --> M2[SVD 분해<br/>U · Σ · V^T]
        M2 --> M3[잠재 요인 추출]
        M3 --> M4[예측 매트릭스 생성]
    end

    U4 --> Final[통합]
    I4 --> Final
    M4 --> Final

    style U2 fill:#e1f5e1
    style I2 fill:#ffe1e1
    style M2 fill:#fff4e1
    style Final fill:#e1f0ff
```

## 컨텐츠 기반 필터링 흐름

```mermaid
graph TB
    subgraph "TF-IDF 필터"
        T1[상품 텍스트<br/>이름+설명+태그] --> T2[TF-IDF<br/>벡터화]
        T2 --> T3[코사인 유사도<br/>계산]
        T3 --> T4[유사 상품 추천]
    end

    subgraph "Feature 기반 필터"
        F1[상품 특성<br/>카테고리/가격/평점] --> F2[특성 벡터 생성<br/>원핫 인코딩]
        F2 --> F3[사용자 선호도<br/>벡터 생성]
        F3 --> F4[유사도 매칭]
    end

    T4 --> Combine[통합 컨텐츠 필터]
    F4 --> Combine

    style T2 fill:#e1f5e1
    style F2 fill:#ffe1e1
    style Combine fill:#e1f0ff
```

## 가중 평균 하이브리드 추천 흐름

```mermaid
graph TB
    Start([사용자 ID 입력]) --> GetProfile[사용자 프로필 조회]

    GetProfile --> CF1[User-based CF<br/>추천 실행]
    GetProfile --> CF2[Item-based CF<br/>추천 실행]
    GetProfile --> CF3[Matrix Factorization<br/>추천 실행]
    GetProfile --> CF4[Content-based<br/>추천 실행]

    CF1 --> N1[점수 정규화<br/>weight=0.2]
    CF2 --> N2[점수 정규화<br/>weight=0.2]
    CF3 --> N3[점수 정규화<br/>weight=0.3]
    CF4 --> N4[점수 정규화<br/>weight=0.3]

    N1 --> Weighted[가중 평균 계산]
    N2 --> Weighted
    N3 --> Weighted
    N4 --> Weighted

    Weighted --> Sort[점수 기준 정렬]
    Sort --> Top[상위 N개 선택]
    Top --> Result([추천 결과 반환])

    style Start fill:#e1f5e1
    style Weighted fill:#fff4e1
    style Result fill:#e1f0ff
```

## 스위칭 하이브리드 추천 흐름

```mermaid
graph TB
    Start([사용자 ID 입력]) --> GetProfile[사용자 프로필 조회]
    GetProfile --> CheckRatings{평가 개수<br/>확인}

    CheckRatings -->|< 5개| ColdStart[Cold Start 상태]
    CheckRatings -->|5-19개| Normal[일반 상태]
    CheckRatings -->|≥ 20개| Active[활발한 상태]

    ColdStart --> UseContent[Content-based<br/>필터 사용]
    Normal --> UseItemCF[Item-based CF<br/>사용]
    Active --> UseMF[Matrix Factorization<br/>사용]

    UseContent --> Result([추천 결과])
    UseItemCF --> Result
    UseMF --> Result

    style Start fill:#e1f5e1
    style CheckRatings fill:#fff4e1
    style Result fill:#e1f0ff
```

## 혼합 하이브리드 추천 흐름

```mermaid
graph TB
    Start([사용자 ID 입력]) --> GetProfile[사용자 프로필 조회]

    GetProfile --> Dist[추천 비율 설정<br/>MF:50%, ItemCF:30%, Content:20%]

    Dist --> MF[Matrix Factorization<br/>5개 추천]
    Dist --> ItemCF[Item-based CF<br/>3개 추천]
    Dist --> Content[Content-based<br/>2개 추천]

    MF --> Dedupe[중복 제거]
    ItemCF --> Dedupe
    Content --> Dedupe

    Dedupe --> Merge[추천 리스트 병합]
    Merge --> Result([10개 추천 결과])

    style Start fill:#e1f5e1
    style Dist fill:#fff4e1
    style Result fill:#e1f0ff
```

## 추천 결과 생성 상세 흐름

```mermaid
graph TB
    Start([추천 요청]) --> Phase1[1단계: 후보 생성]

    Phase1 --> Generate{각 모델에서<br/>후보 생성}

    Generate --> CF[협업 필터링<br/>후보 30개]
    Generate --> CB[컨텐츠 기반<br/>후보 30개]

    CF --> Phase2[2단계: 점수 계산]
    CB --> Phase2

    Phase2 --> Normalize[정규화<br/>0~1 스케일]
    Normalize --> Weight[가중치 적용]

    Weight --> Phase3[3단계: 필터링]
    Phase3 --> RemoveSeen[이미 평가한<br/>상품 제외]
    RemoveSeen --> RemoveLowScore[낮은 점수<br/>상품 제외]

    RemoveLowScore --> Phase4[4단계: 후처리]
    Phase4 --> Sort[점수 기준 정렬]
    Sort --> Diversity[다양성 조정]

    Diversity --> Final[최종 N개 선택]
    Final --> Result([추천 완료])

    style Start fill:#e1f5e1
    style Phase1 fill:#fff4e1
    style Phase2 fill:#ffe1e1
    style Phase3 fill:#e1ffe1
    style Phase4 fill:#e1f0ff
    style Result fill:#e1e1ff
```

## 시스템 실행 시퀀스

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Main as Main System
    participant Data as Dataset
    participant Hybrid as Hybrid Engine
    participant CF as Collaborative Filter
    participant CB as Content Filter

    User->>Main: 추천 요청 (user_id)
    Main->>Data: 사용자 정보 조회
    Data-->>Main: 사용자 프로필

    Main->>Hybrid: 추천 실행
    Hybrid->>CF: 협업 필터링 추천
    CF-->>Hybrid: CF 결과

    Hybrid->>CB: 컨텐츠 기반 추천
    CB-->>Hybrid: CB 결과

    Hybrid->>Hybrid: 결과 통합 및 정렬
    Hybrid-->>Main: 최종 추천 리스트

    Main->>Data: 상품 정보 조회
    Data-->>Main: 상품 상세 정보

    Main-->>User: 추천 결과 표시
```

## 성능 최적화 흐름

```mermaid
graph LR
    subgraph "사전 계산"
        A[시스템 시작] --> B[유사도 매트릭스<br/>사전 계산]
        B --> C[TF-IDF 벡터<br/>사전 계산]
        C --> D[SVD 분해<br/>사전 계산]
    end

    subgraph "런타임 최적화"
        E[추천 요청] --> F{캐시 확인}
        F -->|Hit| G[캐시된 결과 반환]
        F -->|Miss| H[실시간 계산]
        H --> I[결과 캐싱]
    end

    D --> E

    style B fill:#e1f5e1
    style C fill:#ffe1e1
    style D fill:#fff4e1
    style G fill:#e1f0ff
```
