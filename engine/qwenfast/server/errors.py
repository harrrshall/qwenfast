"""One error shape for the whole endpoint: OpenAI's ``{"error": {...}}`` envelope.

Why this module exists. FastAPI's default ``HTTPException`` handler renders
``{"detail": "..."}``. Every OpenAI SDK -- which is the *only* client this API
promises to work with -- reads ``response.json()["error"]["message"]`` and
raises a typed exception from ``error.code``; against ``{"detail": ...}`` it
falls back to a generic "Error code: 400" with the body stringified, or with
nothing at all. The result is rejected requests whose callers cannot see *why*
they were rejected, and a chat page that shows "Engine unavailable." for a
conversation that has merely grown past the context window.

So every refusal from this server -- including the ones FastAPI/Starlette
generate on their own (schema validation, malformed JSON, an unmatched route)
-- goes out as::

    {"error": {"message": str, "type": str, "param": str|null, "code": str|null}}

``type`` follows OpenAI's vocabulary (``invalid_request_error``,
``rate_limit_error``, ``server_error``), and ``code`` is the machine-readable
discriminator SDKs branch on (``context_length_exceeded``, ``rate_limit_exceeded``,
...). ``param`` names the offending request field where there is one, so a
client can point at it.

:class:`ApiError` additionally carries ``error_class`` -- the *metering* label
(``usage.ERROR_CLASSES``). It never appears on the wire; it is what makes
``/admin/usage`` able to say "47 prompt_too_long, 9 context_length_exceeded"
instead of "57 errors".
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

#: status code -> (OpenAI `type`, default `code`, default metering class).
_BY_STATUS: dict[int, tuple[str, Optional[str], str]] = {
    400: ("invalid_request_error", None, "invalid_request"),
    401: ("invalid_request_error", "invalid_api_key", "unauthorized"),
    403: ("invalid_request_error", "permission_denied", "forbidden"),
    404: ("invalid_request_error", "not_found", "invalid_request"),
    405: ("invalid_request_error", "method_not_allowed", "invalid_request"),
    413: ("invalid_request_error", "payload_too_large", "payload_too_large"),
    422: ("invalid_request_error", None, "invalid_request"),
    429: ("rate_limit_error", "rate_limit_exceeded", "rate_limited"),
    500: ("server_error", "internal_error", "invalid_request"),
    503: ("server_error", "service_unavailable", "engine_unavailable"),
    504: ("server_error", "timeout", "request_timeout"),
}


def error_body(
    message: str,
    *,
    type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> dict[str, Any]:
    """The wire envelope. All four fields are always present (SDKs index them)."""
    return {"error": {"message": message, "type": type, "param": param, "code": code}}


class ApiError(HTTPException):
    """An ``HTTPException`` that knows its OpenAI ``type``/``code``/``param``.

    Raised exactly like ``HTTPException``; :func:`install_error_handlers` turns
    it into the envelope above. Plain ``HTTPException``s raised anywhere else
    (including by Starlette itself) still come out in the same shape, using the
    :data:`_BY_STATUS` defaults -- there is no second error format to remember.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: Optional[str] = None,
        code: Optional[str] = None,
        param: Optional[str] = None,
        error_class: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        default_type, default_code, default_class = _BY_STATUS.get(
            status_code, ("invalid_request_error", None, "invalid_request")
        )
        self.error_type = type or default_type
        self.error_code = code if code is not None else default_code
        self.error_param = param
        #: `usage.ERROR_CLASSES` label for the metering row. Never sent to the client.
        self.error_class = error_class or default_class

    @property
    def message(self) -> str:
        return str(self.detail)

    def body(self) -> dict[str, Any]:
        return error_body(
            self.message, type=self.error_type, code=self.error_code, param=self.error_param
        )


def error_class_of(exc: BaseException, status_code: int) -> str:
    """The metering label for any exception that produced `status_code`."""
    cls = getattr(exc, "error_class", None)
    if cls:
        return str(cls)
    return _BY_STATUS.get(status_code, ("", None, "invalid_request"))[2]


def _param_from_validation(exc: RequestValidationError) -> Optional[str]:
    """Best-effort ``param`` for a schema failure: the first offending field.

    ``loc`` looks like ``("body", "messages", 3, "role")``; the leading
    ``"body"`` is noise to a caller who only ever sent a body, so it is dropped
    and the rest joined with dots -- ``messages.3.role``, which is what the
    field is called in the JSON they sent.
    """
    try:
        first = exc.errors()[0]
    except Exception:  # noqa: BLE001
        return None
    loc = [str(p) for p in first.get("loc", ()) if str(p) != "body"]
    return ".".join(loc) or None


def _validation_message(exc: RequestValidationError) -> str:
    try:
        errors = exc.errors()
    except Exception:  # noqa: BLE001
        return "invalid request body"
    if not errors:
        return "invalid request body"
    first = errors[0]
    where = _param_from_validation(exc)
    msg = str(first.get("msg", "invalid value"))
    # Malformed JSON arrives as a validation error with no field path; say so
    # in the words the caller will recognise rather than "Expecting value".
    if first.get("type") in ("json_invalid", "value_error.jsondecode"):
        return f"invalid JSON in request body: {msg}"
    extra = f" ({len(errors) - 1} more problem(s))" if len(errors) > 1 else ""
    return (f"{where}: {msg}" if where else msg) + extra


def install_error_handlers(app: FastAPI, *, on_error=None) -> None:
    """Render every failure in the OpenAI envelope.

    ``on_error(request, status_code, error_class)`` is called only for the
    **schema/JSON** failures, which are the ones nothing else counts: they are
    rejected by FastAPI before any handler runs, so before this they left no
    trace at all -- a client sending malformed bodies all day was invisible in
    exactly the view that exists to find such clients.

    It is deliberately *not* called for ``HTTPException``s: those either come
    from a handler (which meters its own row, with the key name and token counts
    attached) or from auth (`usage.auth_failures`). Metering them here as well
    would double-count every refusal and inflate `totals.requests` with rows
    that have no key and no tokens.
    """

    def _note(request, status_code: int, error_class: str) -> None:
        if on_error is None:
            return
        try:
            on_error(request, status_code, error_class)
        except Exception:  # noqa: BLE001 - metering never fails a response
            pass

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc: StarletteHTTPException):  # noqa: ANN001
        status = exc.status_code
        if isinstance(exc, ApiError):
            body = exc.body()
        else:
            etype, code, _klass = _BY_STATUS.get(
                status, ("invalid_request_error" if status < 500 else "server_error", None,
                         "invalid_request")
            )
            detail = exc.detail
            message = detail if isinstance(detail, str) else str(detail)
            body = error_body(message, type=etype, code=code)
        return JSONResponse(body, status_code=status, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc: RequestValidationError):  # noqa: ANN001
        # 400, not FastAPI's default 422: OpenAI answers a bad body with 400,
        # and the SDK's `BadRequestError` is what a caller is set up to catch.
        _note(request, 400, "invalid_request")
        return JSONResponse(
            error_body(
                _validation_message(exc),
                type="invalid_request_error",
                code="invalid_request_body",
                param=_param_from_validation(exc),
            ),
            status_code=400,
        )


__all__ = ["ApiError", "error_body", "error_class_of", "install_error_handlers"]
