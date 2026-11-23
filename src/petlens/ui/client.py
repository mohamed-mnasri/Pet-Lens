"""A small client for the PetLens API. The pages only talk to the API, never the database."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class ApiClient:
    def __init__(
        self, base_url: str = DEFAULT_URL, http: httpx.Client | None = None, timeout: float = 120
    ) -> None:
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def _json(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                detail = str(r.json().get("detail", r.text))
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, detail)
        return r.json()

    def health(self) -> bool:
        try:
            return bool(self._json(self.http.get("/health")).get("status") == "ok")
        except (httpx.HTTPError, ApiError):
            return False

    def limits(self) -> dict[str, float]:
        result: dict[str, float] = self._json(self.http.get("/limits"))
        return result

    def predict(self, kind: str, name: str, data: bytes) -> dict[str, Any]:
        endpoint = "/predict/image" if kind == "photo" else "/predict/video"
        result: dict[str, Any] = self._json(self.http.post(endpoint, files={"file": (name, data)}))
        return result

    def run(self, upload_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._json(self.http.get(f"/runs/{upload_id}"))
        return result

    def jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._json(self.http.get("/jobs", params={"limit": limit}))
        return result

    def queue_review_batch(self, folder: str) -> dict[str, Any]:
        result: dict[str, Any] = self._json(
            self.http.post("/jobs/review-batch", json={"folder": folder})
        )
        return result

    def review_next(self, queue: str | None = None) -> dict[str, Any] | None:
        params = {"queue": queue} if queue else {}
        result: dict[str, Any] | None = self._json(self.http.get("/review/next", params=params))
        return result

    def review_answer(
        self, card_id: str, answer: str, label: str | None = None, review_ms: int | None = None
    ) -> dict[str, Any]:
        body = {"answer": answer, "label": label, "review_ms": review_ms}
        result: dict[str, Any] = self._json(self.http.post(f"/review/{card_id}", json=body))
        return result

    def review_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = self._json(self.http.get("/review/stats"))
        return result

    def image(self, url: str) -> bytes:
        r = self.http.get(url)
        if r.status_code >= 400:
            raise ApiError(r.status_code, "image not found")
        return r.content
