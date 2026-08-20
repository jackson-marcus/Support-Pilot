"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from supportpilot import __version__
from supportpilot.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="supportpilot",
        description="Support-ticket triage agent: multi-label classification, priority scoring, similar-ticket retrieval, and grounded reply drafting over a product knowledge base.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
