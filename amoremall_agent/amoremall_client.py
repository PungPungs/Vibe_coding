import requests


class AmoreMallClient:
    BASE_URL = "https://www.amoremall.com"
    CATEGORIES_API = "https://api-gw.amoremall.com/display/v2/M01/menus/categories"
    REVIEWS_API = "https://api-gw.amoremall.com/commune/v2/M01/apcp/reviews"
    PRODUCTS_API = "https://api-gw.amoremall.com/display/v2/M01/online-products/by-category"

    def _default_reviews(
        self,
        onlineProdSn: int,
        offset: int = 0,
        OnlineProd: str = "OnlineProd",
        prodReviewType: str = "All",
        BestScrOnly: str = "BestScrOnly",
        Scope: str = "All",
        filterMemberAttrYn: str = "N",
        limit: int = 100,
        imageOnlyYn: str = "N",
    ) -> dict:
        return {
            "onlineProdSn": onlineProdSn,
            "offset": offset,
            "prodReviewUnit": OnlineProd,
            "prodReviewType": prodReviewType,
            "prodReviewSort": BestScrOnly,
            "scope": Scope,
            "filterMemberAttrYn": filterMemberAttrYn,
            "limit": limit,
            "imageOnlyYn": imageOnlyYn,
        }

    def _default_products(
        self,
        categorySn: int,
        containsFilter: str = "True",
        limit: int = 100,
        offset: int = 0,
        sortType: str = "Ranking",
    ) -> dict:
        return {
            "categorySn": categorySn,
            "containsFilter": containsFilter,
            "limit": limit,
            "offset": offset,
            "sortType": sortType,
        }

    def get_reviews(self, onlineProdSn: int, **kwargs) -> dict | bool:
        """onlineProdSn를 통한 리뷰 추출"""
        params = self._default_reviews(onlineProdSn=onlineProdSn, **kwargs)
        res = requests.get(self.REVIEWS_API, params=params, timeout=30)
        if res.status_code == 200:
            return res.json()
        return False

    def get_categories(self) -> dict | bool:
        """카테고리 추출"""
        res = requests.get(self.CATEGORIES_API, timeout=30)
        if res.status_code == 200:
            return res.json()
        return False

    def get_products(self, categorySn: int, **kwargs) -> dict | bool:
        """categorySn를 통한 제품 추출"""
        params = self._default_products(categorySn=categorySn, **kwargs)
        res = requests.get(self.PRODUCTS_API, params=params, timeout=30)
        if res.status_code == 200:
            return res.json()
        return False
