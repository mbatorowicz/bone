"""ASGI Studio — ten sam kontrakt HTTP co lokalny serwer, hostowany na Vercelu.

Vercel ładuje ``app`` przez ``[tool.vercel] entrypoint``. Silnik dalej biegnie
w wątku tła; odpytywanie ``/api/view`` trzyma izolat przy życiu (Fluid Compute).
Na Hobby limity cząstek i ``/tmp`` nakłada ``apply_host_limits``.
"""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response

from bone.studio.server import ApiResponse, WEB_ROOT, dispatch_get, dispatch_post

app = FastAPI(title="Bone Studio", docs_url=None, redoc_url=None)


def _asgi_response(result: ApiResponse) -> Response:
    return Response(
        content=result.body,
        status_code=result.status,
        headers={
            "Content-Type": result.content_type,
            "Cache-Control": "no-store",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"ok": "true", "web": str(WEB_ROOT.is_dir())}


@app.api_route("/", methods=["GET"])
@app.api_route("/{full_path:path}", methods=["GET", "POST"])
async def studio(request: Request, full_path: str = "") -> Response:
    path = "/" if not full_path else f"/{full_path}"
    # Starlette czasem zostawia trailing slash przy mountach — normalizujemy
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = urlparse(str(request.url)).query

    if request.method == "GET":
        return _asgi_response(dispatch_get(path, query))

    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        return _asgi_response(
            ApiResponse.json(400, {"error": "ciało żądania nie jest poprawnym JSON-em"})
        )
    return _asgi_response(dispatch_post(path, body))
