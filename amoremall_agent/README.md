# 아모레몰 CRM 메시지 생성 Agent

아모레몰 API/Vector DB(pgvector)를 활용해 고객 페르소나에 맞는 CRM 메시지를 생성하는 Streamlit 앱입니다.

## 주요 기능
- 고객 페르소나/정보/발신 목적 기반 메시지 생성
- CSV 대량 입력 지원
- 브랜드 톤 저장 및 관리
- pgvector 기반 RAG 검색 (제품/리뷰/브랜드 톤)
- AmoreMallClient로 카테고리/제품/리뷰 조회 및 DB 적재

## 실행 방법
```bash
cd amoremall_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 환경 변수
- PostgreSQL DSN은 기본값을 사용하거나 사이드바에서 입력합니다.
- pgvector 확장과 DB는 이미 Docker에서 구동 중이어야 합니다.

## CSV 입력 컬럼
`persona`, `customer_info`, `purpose`, `brand`, `channel`

## 폴더 구조
```
 amoremall_agent/
 ├── app.py
 ├── amoremall_client.py
 ├── rag.py
 ├── message_generator.py
 └── requirements.txt
```
