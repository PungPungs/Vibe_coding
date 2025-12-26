from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class MessageContext:
    persona: str
    customer_info: str
    purpose: str
    brand: str
    channel: str
    product_name: str | None = None
    product_description: str | None = None
    brand_tone: str | None = None
    review_snippets: list[str] | None = None


def _join_nonempty(parts: Iterable[str | None], sep: str = " ") -> str:
    return sep.join([part for part in parts if part])


def generate_message(context: MessageContext) -> str:
    tone = context.brand_tone or "브랜드 감성과 신뢰감을 담아 자연스럽게 전달합니다."
    reviews = context.review_snippets or []
    review_sentence = ""
    if reviews:
        review_sentence = f"실사용 후기는 '{reviews[0]}'처럼 만족도가 높아요."

    product_line = ""
    if context.product_name:
        product_line = f"오늘은 {context.product_name}을(를) 추천드려요."
    if context.product_description:
        product_line = _join_nonempty([product_line, context.product_description], sep=" ")

    intro = (
        f"{context.customer_info} 고객님께, {context.persona} 취향을 고려해"
        f" {context.purpose}에 맞는 큐레이션을 준비했어요."
    )

    closing = "지금 바로 아모레몰에서 만나보세요."

    if context.channel == "SMS":
        closing = "지금 바로 확인해보세요."
    elif context.channel == "카카오톡":
        closing = "카카오톡에서 바로 만나보세요."

    return "\n".join(
        [
            intro,
            tone,
            product_line,
            review_sentence,
            closing,
        ]
    ).strip()
