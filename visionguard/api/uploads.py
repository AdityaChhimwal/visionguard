"""Bound request bytes and read time before invoking any image decoder."""

import asyncio

from fastapi import Request


class UploadProblem(ValueError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


async def read_upload(request: Request, limit: int, timeout: int) -> bytes:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
            if declared < 0:
                raise ValueError
        except ValueError as exc:
            raise UploadProblem(400, "invalid_length", "Invalid upload length.") from exc
        if declared > limit:
            raise UploadProblem(413, "byte_limit", "This file exceeds the upload limit.")
    data = bytearray()
    try:
        async with asyncio.timeout(timeout):
            async for part in request.stream():
                if len(part) + len(data) > limit:
                    raise UploadProblem(413, "byte_limit", "This file exceeds the upload limit.")
                data.extend(part)
    except TimeoutError as exc:
        raise UploadProblem(408, "upload_timeout", "Upload timed out. Please try again.") from exc
    if not data:
        raise UploadProblem(400, "empty_upload", "Choose an image to analyze.")
    return bytes(data)
