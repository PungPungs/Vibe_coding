from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from amoremall_client import AmoreMallClient
from message_generator import MessageContext, generate_message
from rag import PgVectorStore


DEFAULT_DSN = "postgresql://postgres:postgres@localhost:54320/postgres"
DEFAULT_BRAND_TONES = {
    "라네즈": "맑고 촉촉한 수분감, 트렌디한 감성을 강조하는 톤",
    "헤라": "모던하고 고급스러운 무드, 자신감을 전달하는 톤",
    "이니스프리": "자연 친화적이고 편안한 감성, 데일리 스킨케어 톤",
    "설화수": "전통과 품격, 깊이 있는 스토리텔링 톤",
}


st.set_page_config(page_title="아모레몰 CRM 메시지 생성", layout="wide")

if "brand_tones" not in st.session_state:
    st.session_state.brand_tones = DEFAULT_BRAND_TONES.copy()


@st.cache_data(show_spinner=False)
def _load_sample_csv() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "persona": "민감하지만 촉촉한 윤광을 선호하는 고객",
                "customer_info": "30대/여성/건성",
                "purpose": "신상품 출시",
                "brand": "라네즈",
                "channel": "SMS",
            },
            {
                "persona": "비건 성분과 지속가능 패키징을 중요하게 보는 고객",
                "customer_info": "20대/여성/복합성",
                "purpose": "할인 상품 출시",
                "brand": "이니스프리",
                "channel": "카카오톡",
            },
        ]
    )


def _connect_store(dsn: str) -> PgVectorStore:
    store = PgVectorStore(dsn)
    store.ensure_schema()
    return store


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_categories(payload: dict[str, Any]) -> list[dict[str, Any]]:
    categories = []
    for item in payload.get("categories", []) + payload.get("data", []):
        if isinstance(item, dict) and "categorySn" in item:
            categories.append(item)
    return categories


def _extract_products(payload: dict[str, Any]) -> list[dict[str, Any]]:
    products = []
    for item in payload.get("products", []) + payload.get("data", []):
        if isinstance(item, dict) and "onlineProdSn" in item:
            products.append(item)
    return products


def _extract_reviews(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = []
    for item in payload.get("reviews", []) + payload.get("data", []):
        if isinstance(item, dict) and "reviewNo" in item:
            reviews.append(item)
    return reviews


def _build_query(persona: str, customer_info: str, purpose: str, brand: str) -> str:
    return " ".join([persona, customer_info, purpose, brand]).strip()


st.title("아모레몰 맞춤형 CRM 메시지 생성")

with st.sidebar:
    st.subheader("Vector DB 설정")
    dsn = st.text_input("PostgreSQL DSN", value=DEFAULT_DSN)
    init_db = st.button("스키마 초기화")
    if init_db:
        try:
            _connect_store(dsn)
            st.success("스키마를 초기화했습니다.")
        except Exception as exc:
            st.error(f"스키마 초기화 실패: {exc}")

    st.subheader("브랜드 톤 관리")
    tone_brand = st.text_input("브랜드명", value="라네즈")
    tone_text = st.text_area("브랜드 톤 설명", value=st.session_state.brand_tones.get(tone_brand, ""))
    if st.button("브랜드 톤 저장"):
        st.session_state.brand_tones[tone_brand] = tone_text
        try:
            store = _connect_store(dsn)
            store.upsert_brand_tones(
                [
                    {
                        "brand": tone_brand,
                        "tone_text": tone_text,
                        "metadata": {"source": "manual"},
                    }
                ]
            )
            st.success("브랜드 톤을 저장했습니다.")
        except Exception as exc:
            st.warning(f"DB 저장 실패(로컬 톤만 적용): {exc}")


form_tab, csv_tab, ingest_tab = st.tabs(["폼 입력", "CSV 입력", "데이터 수집/관리"])

with form_tab:
    st.subheader("개별 고객 메시지 생성")
    col1, col2, col3 = st.columns(3)

    with col1:
        persona = st.selectbox(
            "고객 페르소나",
            [
                "수분 케어를 중시하는 실용파",
                "트렌디한 메이크업을 즐기는 고객",
                "저자극 성분을 선호하는 민감성 고객",
                "자연 친화적 라이프스타일을 추구하는 고객",
                "커스텀 입력",
            ],
        )
        if persona == "커스텀 입력":
            persona = st.text_input("페르소나 상세", value="촉촉한 윤광을 선호하는 고객")

    with col2:
        customer_info = st.selectbox(
            "고객 정보",
            [
                "20대/여성/지성",
                "30대/여성/건성",
                "40대/남성/복합성",
                "30대/남성/지성",
                "커스텀 입력",
            ],
        )
        if customer_info == "커스텀 입력":
            customer_info = st.text_input("고객 정보 상세", value="30대/여성/건성")

    with col3:
        purpose = st.selectbox(
            "발신 목적",
            ["신상품 출시", "할인 상품 출시", "리뷰 이벤트 안내", "재구매 유도", "샘플링 제안"],
        )
        channel = st.selectbox("발신 채널", ["SMS", "카카오톡", "이메일"])

    brand = st.selectbox("브랜드 선택", list(st.session_state.brand_tones.keys()))
    generate = st.button("메시지 생성")

    if generate:
        try:
            store = _connect_store(dsn)
            query = _build_query(persona, customer_info, purpose, brand)
            product_hits = store.search(query, "amoremall_products", "description", top_k=3)
            brand_hits = store.search(query, "amoremall_brand_tones", "tone_text", top_k=1)
            review_hits = store.search(query, "amoremall_reviews", "review_text", top_k=2)
        except Exception as exc:
            st.warning(f"RAG 검색 실패: {exc}")
            product_hits, brand_hits, review_hits = [], [], []

        product_name = None
        if product_hits:
            product_name = product_hits[0].metadata.get("name") or product_hits[0].metadata.get(
                "onlineProdName"
            )
        product_desc = product_hits[0].content if product_hits else None
        brand_tone = brand_hits[0].content if brand_hits else st.session_state.brand_tones.get(brand, "")
        review_snippets = [hit.content for hit in review_hits]

        context = MessageContext(
            persona=persona,
            customer_info=customer_info,
            purpose=purpose,
            brand=brand,
            channel=channel,
            product_name=product_name,
            product_description=product_desc,
            brand_tone=brand_tone,
            review_snippets=review_snippets,
        )

        message = generate_message(context)
        st.text_area("생성 메시지", message, height=220)

        if product_hits:
            st.caption("추천 근거 - 제품")
            for hit in product_hits:
                st.write(f"- {hit.metadata.get('name', '제품')} (점수 {hit.score:.2f})")

        if review_hits:
            st.caption("추천 근거 - 리뷰")
            for hit in review_hits:
                st.write(f"- {hit.content}")

with csv_tab:
    st.subheader("다중 고객 CSV 처리")
    sample_df = _load_sample_csv()
    st.download_button(
        "샘플 CSV 다운로드",
        sample_df.to_csv(index=False).encode("utf-8"),
        file_name="amoremall_sample.csv",
        mime="text/csv",
    )

    upload = st.file_uploader("CSV 업로드", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
        st.write("입력 데이터", df)

        if st.button("CSV 메시지 생성"):
            results = []
            for _, row in df.iterrows():
                persona = str(row.get("persona", ""))
                customer_info = str(row.get("customer_info", ""))
                purpose = str(row.get("purpose", ""))
                brand = str(row.get("brand", ""))
                channel = str(row.get("channel", "SMS"))

                try:
                    store = _connect_store(dsn)
                    query = _build_query(persona, customer_info, purpose, brand)
                    product_hits = store.search(query, "amoremall_products", "description", top_k=1)
                    brand_hits = store.search(query, "amoremall_brand_tones", "tone_text", top_k=1)
                    review_hits = store.search(query, "amoremall_reviews", "review_text", top_k=1)
                except Exception:
                    product_hits, brand_hits, review_hits = [], [], []

                product_name = None
                if product_hits:
                    product_name = product_hits[0].metadata.get("name") or product_hits[0].metadata.get(
                        "onlineProdName"
                    )

                context = MessageContext(
                    persona=persona,
                    customer_info=customer_info,
                    purpose=purpose,
                    brand=brand,
                    channel=channel,
                    product_name=product_name,
                    product_description=product_hits[0].content if product_hits else None,
                    brand_tone=brand_hits[0].content
                    if brand_hits
                    else st.session_state.brand_tones.get(brand, ""),
                    review_snippets=[hit.content for hit in review_hits],
                )
                results.append({"message": generate_message(context), **row.to_dict()})

            result_df = pd.DataFrame(results)
            st.write("결과", result_df)
            st.download_button(
                "결과 다운로드",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="amoremall_messages.csv",
                mime="text/csv",
            )

with ingest_tab:
    st.subheader("API 데이터 수집")
    client = AmoreMallClient()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("카테고리 조회"):
            payload = client.get_categories()
            if payload:
                st.json(payload)
            else:
                st.error("카테고리 조회 실패")

    with col2:
        category_sn = st.number_input("카테고리 번호", min_value=1, value=1)
        if st.button("제품 조회"):
            payload = client.get_products(int(category_sn))
            if payload:
                st.json(payload)
            else:
                st.error("제품 조회 실패")

    st.divider()
    st.subheader("DB 적재")
    category_sn = st.number_input("제품 적재용 카테고리 번호", min_value=1, value=1, key="ingest_category")
    if st.button("제품 적재"):
        payload = client.get_products(int(category_sn))
        if payload:
            items = _extract_products(payload)
            rows = []
            for item in items:
                rows.append(
                    {
                        "product_id": str(item.get("onlineProdSn")),
                        "name": item.get("onlineProdName") or item.get("name"),
                        "brand": item.get("brandName"),
                        "category": item.get("categoryName"),
                        "description": item.get("summary") or item.get("prodSummary"),
                        "metadata": item,
                    }
                )
            try:
                store = _connect_store(dsn)
                store.upsert_products(rows)
                st.success(f"{len(rows)}개 제품을 적재했습니다.")
            except Exception as exc:
                st.error(f"적재 실패: {exc}")
        else:
            st.error("제품 조회 실패")

    online_prod_sn = st.number_input("리뷰 적재용 상품 번호", min_value=1, value=1, key="review_prod")
    if st.button("리뷰 적재"):
        payload = client.get_reviews(int(online_prod_sn))
        if payload:
            items = _extract_reviews(payload)
            rows = []
            for item in items:
                rows.append(
                    {
                        "review_id": str(item.get("reviewNo")),
                        "product_id": str(online_prod_sn),
                        "review_text": item.get("reviewText") or item.get("reviewCont"),
                        "metadata": item,
                    }
                )
            try:
                store = _connect_store(dsn)
                store.upsert_reviews(rows)
                st.success(f"{len(rows)}개 리뷰를 적재했습니다.")
            except Exception as exc:
                st.error(f"리뷰 적재 실패: {exc}")
        else:
            st.error("리뷰 조회 실패")

    st.divider()
    st.subheader("현재 등록된 브랜드 톤")
    st.json(st.session_state.brand_tones)
