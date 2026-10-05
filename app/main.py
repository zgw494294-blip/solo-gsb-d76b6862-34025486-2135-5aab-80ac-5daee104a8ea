from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from . import registry
from .engine import (
    ValidationError,
    compare,
    compose,
    preview,
    prune,
    rename,
    review,
    revise,
)
from .models import (
    CompareRequest,
    ComposeRequest,
    GraphPreviewRequest,
    GraphRegisterRequest,
    GraphRevisionRequest,
    PackagePurgeRequest,
    PackageReleaseRequest,
    PreviewRequest,
    PruneRequest,
    RenameRequest,
    ReviewRequest,
    ReviseRequest,
    SensitiveColumnsReplaceRequest,
)
from .storage import StoreHolder


_DESCRIPTION = (
    "Reviews every output column of an export node for sensitive content "
    "provenance and sensitive row-level influence, can revise a blocked "
    "graph into a re-submittable, review-safe export, can prune a graph "
    "down to a chosen subset of export columns, can batch-rename "
    "columns of a source node while keeping every in-graph reference "
    "pointing at the same original columns, can compose an upstream "
    "graph's export into a downstream graph's source position, yielding "
    "one reviewable combined graph, can run a small-sample preview "
    "of an export over caller-provided source rows and filter "
    "conditions, returning sample data only when the export passes "
    "review, and can diff a baseline graph against a candidate graph "
    "to report the export risks the candidate newly introduces. "
    "Graphs can additionally be registered by id and reviewed/previewed "
    "against a shared, versioned sensitive-column registry that is ORed "
    "with every graph's in-graph sensitive marks. A registered graph can "
    "be revised under the same id: each accepted submission appends a new "
    "graph_version (an optimistic expected_version guards against lost "
    "updates); reads and by-id previews always use the latest version, "
    "while every sealed package stays bound to the version sealed. A "
    "registered graph's passing small-sample result can be sealed into a "
    "pending-release export package, released under the current registry "
    "generation (the release re-reviews the package's bound historical "
    "version, not the current head) and claimed afterwards — release and "
    "claim always serve the sealed data, never a different batch. Sealed "
    "samples can finally be purged by source column: the dependency "
    "decision runs against the bound historical graph under the original "
    "content and row-influence rules, ignoring sensitivity marks, and "
    "every matching live package loses its stored rows and release record "
    "and stays purged for good — release and claim then answer 410. "
    "Committed registry replacements and sample purges are appended to a "
    "seq-ordered audit log (stored in the same SQLite database, in the "
    "same transactions as the changes they describe) that can be paged "
    "through via GET /api/audit-log."
)


def create_app(db_path: Optional[str] = None) -> FastAPI:
    """Build an application instance owning its own SQLite storage.

    ``db_path`` selects the database for this instance only: an explicit
    path always wins; otherwise the path is resolved from ``RISK_API_DB``
    (with the existing ``data/export-risk.db`` default) when the instance
    starts. The instance opens its connection in the lifespan startup and
    closes it at shutdown. Two instances created in the same process with
    different paths keep fully independent connections, locks, graphs,
    registry generations, packages and audit logs, and shutting one down
    never closes the other's connection.
    """
    holder = StoreHolder(db_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Open (and, if needed, create) the SQLite database eagerly so a
        # missing or unwritable volume is visible at startup; this instance
        # owns exactly this connection for its whole lifetime.
        holder.open()
        try:
            yield
        finally:
            holder.close()

    application = FastAPI(
        title="Local Data Export Risk Review API",
        description=_DESCRIPTION,
        version="1.11.0",
        lifespan=lifespan,
    )
    application.state.store_holder = holder
    _register_routes(application)
    return application


def _register_routes(app: FastAPI) -> None:
    @app.exception_handler(ValidationError)
    async def _validation_error_handler(
        _request: Request, exc: ValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": exc.detail()})

    @app.exception_handler(Exception)
    async def _unexpected_error_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        # Never leak stack traces from a local review service.
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL", "message": str(exc)}},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/review")
    async def review_export(request: ReviewRequest) -> dict[str, object]:
        # Legacy review stays driven by in-graph sensitive marks only.
        return review(request)

    @app.post("/api/revise")
    async def revise_export(request: ReviseRequest) -> dict[str, object]:
        return revise(request)

    @app.post("/api/prune")
    async def prune_export(request: PruneRequest) -> dict[str, object]:
        return prune(request)

    @app.post("/api/rename")
    async def rename_source_columns(request: RenameRequest) -> dict[str, object]:
        return rename(request)

    @app.post("/api/compose")
    async def compose_graphs(request: ComposeRequest) -> dict[str, object]:
        return compose(request)

    @app.post("/api/preview")
    async def preview_export(request: PreviewRequest) -> dict[str, object]:
        # Legacy preview stays gated on in-graph marks only.
        return preview(request)

    @app.post("/api/compare")
    async def compare_exports(request: CompareRequest) -> dict[str, object]:
        return compare(request)

    # ------------------------------------------------------------------
    # Registered graphs and the shared sensitive-column registry
    # ------------------------------------------------------------------

    @app.post("/api/graphs", status_code=201)
    async def register_graph(
        http_request: Request, request: GraphRegisterRequest
    ) -> dict[str, object]:
        # Store the original submission verbatim (FastAPI already parsed and
        # schema-validated it); the engine re-runs the whole-graph validation.
        raw_nodes = [
            node.model_dump(by_alias=True, exclude_unset=True)
            for node in request.nodes  # type: ignore[attr-defined]
        ]
        db = http_request.app.state.store_holder.get()
        return registry.register_graph(db, raw_nodes, request.export)

    @app.put("/api/graphs/{graph_id}")
    async def revise_registered_graph(
        http_request: Request,
        graph_id: str,
        request: GraphRevisionRequest,
    ) -> object:
        # A revision stores the new submission verbatim under the same id; the
        # engine re-runs the whole-graph validation and the registry layer
        # checks the optimistic expected_version. A blocked graph is still
        # saved.
        raw_nodes = [
            node.model_dump(by_alias=True, exclude_unset=True)
            for node in request.nodes  # type: ignore[attr-defined]
        ]
        db = http_request.app.state.store_holder.get()
        try:
            return registry.revise_graph(
                db, graph_id, raw_nodes, request.export,
                request.expected_version,
            )
        except registry.GraphNotFound:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "GRAPH_NOT_FOUND",
                                   "message": f"no registered graph with id '{graph_id}'"}},
            )
        except registry.StaleVersion as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "VERSION_CONFLICT",
                        "message": str(exc),
                        "expected_version": exc.expected,
                        "current_version": exc.current,
                    }
                },
            )

    @app.get("/api/graphs/{graph_id}")
    async def read_graph(http_request: Request, graph_id: str) -> object:
        db = http_request.app.state.store_holder.get()
        try:
            return registry.read_graph(db, graph_id)
        except registry.GraphNotFound:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "GRAPH_NOT_FOUND",
                                   "message": f"no registered graph with id '{graph_id}'"}},
            )

    @app.post("/api/graphs/{graph_id}/preview")
    async def preview_registered_graph(
        http_request: Request,
        graph_id: str,
        request: GraphPreviewRequest,
    ) -> object:
        db = http_request.app.state.store_holder.get()
        try:
            return registry.preview_graph(
                db, graph_id, request.rows, request.conditions
            )
        except registry.GraphNotFound:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "GRAPH_NOT_FOUND",
                                   "message": f"no registered graph with id '{graph_id}'"}},
            )

    @app.post("/api/graphs/{graph_id}/packages", status_code=201)
    async def seal_graph_package(
        http_request: Request,
        graph_id: str,
        request: GraphPreviewRequest,
    ) -> object:
        db = http_request.app.state.store_holder.get()
        try:
            return registry.seal_package(
                db, graph_id, request.rows, request.conditions
            )
        except registry.GraphNotFound:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "GRAPH_NOT_FOUND",
                                   "message": f"no registered graph with id '{graph_id}'"}},
            )

    @app.post("/api/packages/{package_id}/release")
    async def release_sealed_package(
        http_request: Request,
        package_id: str,
        request: PackageReleaseRequest,
    ) -> object:
        db = http_request.app.state.store_holder.get()
        try:
            return registry.release_package(
                db, package_id, request.expected_generation
            )
        except registry.PackageNotFound:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "PACKAGE_NOT_FOUND",
                                   "message": f"no sealed package with id '{package_id}'"}},
            )
        except registry.PackagePurged as exc:
            return JSONResponse(
                status_code=410,
                content={"error": {"code": "PACKAGE_PURGED", "message": str(exc)}},
            )
        except registry.StaleGeneration as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "GENERATION_CONFLICT",
                        "message": str(exc),
                        "expected_generation": exc.expected,
                        "current_generation": exc.current,
                    }
                },
            )

    @app.get("/api/packages/{package_id}")
    async def claim_sealed_package(
        http_request: Request, package_id: str
    ) -> object:
        db = http_request.app.state.store_holder.get()
        try:
            return registry.claim_package(db, package_id)
        except registry.PackageNotFound:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "PACKAGE_NOT_FOUND",
                                   "message": f"no sealed package with id '{package_id}'"}},
            )
        except registry.PackagePurged as exc:
            return JSONResponse(
                status_code=410,
                content={"error": {"code": "PACKAGE_PURGED", "message": str(exc)}},
            )
        except registry.PackageNotReleased as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "PACKAGE_NOT_RELEASED",
                        "message": str(exc),
                        "release_generation": exc.release_generation,
                        "current_generation": exc.current,
                    }
                },
            )

    @app.post("/api/packages/purge")
    async def purge_sealed_packages(
        http_request: Request, request: PackagePurgeRequest
    ) -> object:
        columns = [(item.source, item.column) for item in request.columns]
        db = http_request.app.state.store_holder.get()
        return registry.purge_packages(db, columns)

    @app.get("/api/sensitive-columns")
    async def read_sensitive_columns(http_request: Request) -> dict[str, object]:
        db = http_request.app.state.store_holder.get()
        return registry.get_registry(db)

    @app.put("/api/sensitive-columns")
    async def replace_sensitive_columns(
        http_request: Request,
        request: SensitiveColumnsReplaceRequest,
    ) -> object:
        entries = [(item.source, item.column) for item in request.entries]
        db = http_request.app.state.store_holder.get()
        try:
            return registry.replace_registry(
                db, entries, request.expected_generation
            )
        except registry.StaleGeneration as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "GENERATION_CONFLICT",
                        "message": str(exc),
                        "expected_generation": exc.expected,
                        "current_generation": exc.current,
                    }
                },
            )

    @app.get("/api/audit-log")
    async def read_audit_log(
        http_request: Request,
        after_seq: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        until_seq: int | None = Query(default=None, ge=0),
    ) -> object:
        """Page through committed registry replacements and sample purges.

        Both operation kinds share one increasing ``seq`` in effect order. The
        page covers ``after_seq < seq <= until_seq`` (ascending); omitting
        ``until_seq`` pins the cutoff to the latest committed seq, while an
        explicit cutoff past that value is rejected as 400.
        """
        db = http_request.app.state.store_holder.get()
        try:
            return registry.read_audit_log(db, after_seq, limit, until_seq)
        except registry.UntilSeqOutOfRange as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "UNTIL_SEQ_OUT_OF_RANGE",
                        "message": str(exc),
                        "until_seq": exc.until_seq,
                        "max_seq": exc.max_seq,
                    }
                },
            )


# Module-level ASGI entry point kept unchanged: ``uvicorn app.main:app`` and
# the Docker single-command startup create this instance, which resolves
# RISK_API_DB (and its default) at lifespan startup.
app = create_app()
