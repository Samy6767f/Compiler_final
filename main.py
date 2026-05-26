import asyncio, json, time, logging, os
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, AsyncGenerator, Optional, Union, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-compiler")

app = FastAPI(title="AI Compiler API", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from pipeline.orchestrator import Pipeline
    pipeline_llm = Pipeline(use_llm=True)
    logger.info("Pipeline loaded OK")
except Exception as e:
    logger.error(f"Pipeline import failed: {e}")
    pipeline_llm = None

class CompileRequest(BaseModel):
    prompt: str

# Stage definitions — keys match what main.py emits
STAGES = [
    {"name": "1_intent_extraction",    "display": "Intent Extraction"},
    {"name": "2_system_design",        "display": "System Design"},
    {"name": "3_schema_generation",    "display": "Schema Generation"},
    {"name": "4_validation_refinement","display": "Validation & Repair"},
    {"name": "5_output",               "display": "Output Ready"},
]
STAGE_PROGRESS = {
    "1_intent_extraction":     20,
    "2_system_design":         40,
    "3_schema_generation":     65,
    "4_validation_refinement": 85,
    "5_output":               100,
}

tracker_store: Dict[str, "ProgressTracker"] = {}


class ProgressTracker:
    def __init__(self, request_id: str):
        self.request_id  = request_id
        self.start_time  = time.time()
        self.is_complete = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._result: Optional[dict] = None

    async def emit(self, stage: str, status: str, details: str = "", error: str = None):
        event = {
            "type":      "stage_update",
            "stage":     stage,
            "status":    status,
            "progress":  STAGE_PROGRESS.get(stage, 0),
            "timestamp": round(time.time() - self.start_time, 2),
            "details":   details,
            "error":     error,
        }
        await self._queue.put(event)
        if status == "completed" and stage == "5_output":
            self.is_complete = True
            await self._queue.put({"type": "complete", "progress": 100})

    async def event_stream(self) -> AsyncGenerator:
        while not self.is_complete or not self._queue.empty():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
        yield 'data: {"type":"close"}\n\n'


@app.get("/")
async def root():
    return FileResponse("www/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline": pipeline_llm is not None}

@app.post("/compile")
async def compile_endpoint(req: CompileRequest, bg: BackgroundTasks):
    if pipeline_llm is None:
        return JSONResponse({"error": "Pipeline unavailable", "success": False}, status_code=503)

    request_id = f"req_{int(time.time()*1000)}"
    tracker = ProgressTracker(request_id)
    tracker_store[request_id] = tracker

    logger.info(f"[{request_id}] compile: prompt_len={len(req.prompt)}")
    bg.add_task(run_pipeline, req.prompt, request_id, tracker)
    return JSONResponse({"request_id": request_id, "success": True})

@app.get("/compile-stream/{request_id}")
async def compile_stream(request_id: str):
    if request_id not in tracker_store:
        return JSONResponse({"error": "Not found"}, status_code=404)
    tracker = tracker_store[request_id]
    return StreamingResponse(
        tracker.event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

@app.get("/compile-result/{request_id}")
async def get_result(request_id: str):
    if request_id not in tracker_store:
        return JSONResponse({"error": "Not found"}, status_code=404)
    tracker = tracker_store[request_id]
    if not tracker.is_complete:
        return JSONResponse({"error": "Still processing"}, status_code=202)
    return JSONResponse(tracker._result or {"error": "No result", "success": False})


def _flatten_validation_errors(raw_errors: Any) -> List[str]:
    """
    Recursively flatten any structure (tuple, list, dict, string) into a list of strings.
    Handles both (errors, warnings) tuple and malformed returns.
    """
    flat = []
    if raw_errors is None:
        return flat
    if isinstance(raw_errors, str):
        return [raw_errors]
    if isinstance(raw_errors, (tuple, list)):
        for item in raw_errors:
            flat.extend(_flatten_validation_errors(item))
        return flat
    if isinstance(raw_errors, dict):
        # If dict, take values (might be errors dict)
        for val in raw_errors.values():
            flat.extend(_flatten_validation_errors(val))
        return flat
    # Fallback: convert to string
    return [str(raw_errors)]


async def run_pipeline(prompt: str, request_id: str, tracker: ProgressTracker):
    """
    Runs the full 5-stage pipeline.
    Result shape matches www/index.html expectations.
    """
    stage_timings: Dict[str, float] = {}
    t0 = time.time()
    result: dict = {}

    def ms(t_start: float) -> float:
        return round((time.time() - t_start) * 1000, 1)

    try:
        # ── Stage 1: Intent ───────────────────────────────────────────────────
        await tracker.emit("1_intent_extraction", "started", "Analyzing prompt...")
        ts = time.time()
        intent = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pipeline_llm.intent_extractor.extract_with_review(prompt)
        )
        stage_timings["1_intent_extraction"] = ms(ts)
        n_ent  = len(intent.get("entities", []))
        n_role = len(intent.get("roles", []))
        await tracker.emit("1_intent_extraction", "completed",
                           f"Found {n_ent} entities, {n_role} roles")

        # ── Stage 2: Design ───────────────────────────────────────────────────
        await tracker.emit("2_system_design", "started", "Designing architecture...")
        ts = time.time()
        design = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pipeline_llm.system_designer.design_llm(intent)
        )
        stage_timings["2_system_design"] = ms(ts)
        await tracker.emit("2_system_design", "completed",
                           f"{len(design.get('pages',[]))} pages, {len(design.get('entities',[]))} entities")

        # ── Stage 3: Schemas ──────────────────────────────────────────────────
        await tracker.emit("3_schema_generation", "started", "Generating DB · API · UI · Auth...")
        ts = time.time()
        schemas = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pipeline_llm.schema_generator.generate_llm(design)
        )
        stage_timings["3_schema_generation"] = ms(ts)
        n_tables = len(schemas.get("db", {}).get("tables", {}))
        n_eps    = len(schemas.get("api", {}).get("endpoints", []))
        await tracker.emit("3_schema_generation", "completed",
                           f"{n_tables} tables, {n_eps} endpoints")

        # ── Stage 4: Validation ───────────────────────────────────────────────
        await tracker.emit("4_validation_refinement", "started", "Validating cross-layer consistency...")
        ts = time.time()
        raw_errors = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pipeline_llm.validator.validate_cross_layer(design, schemas)
        )
        stage_timings["4_validation_refinement"] = ms(ts)

        # Flatten any nested structure into a list of strings
        flat_errors = _flatten_validation_errors(raw_errors)

        # Separate issues (critical) from notes (warnings/info)
        keywords = ("undefined", "missing", "no ", "invalid", "failed", "mismatch")
        issues = []
        notes = []
        for err in flat_errors:
            if not isinstance(err, str):
                err = str(err)
            if any(k in err.lower() for k in keywords):
                issues.append(err)
            else:
                notes.append(err)

        await tracker.emit("4_validation_refinement", "completed",
                           "Passed" if not issues else f"{len(issues)} issues found")

        # ── Stage 5: Simulation ───────────────────────────────────────────────
        await tracker.emit("5_output", "started", "Running execution simulation...")
        ts = time.time()
        simulation = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pipeline_llm.simulator.simulate_execution(schemas)
        )
        # Merge validation issues into simulation fails so UI shows them
        simulation.setdefault("checks_failed", [])
        if isinstance(simulation["checks_failed"], list):
            simulation["checks_failed"].extend(issues)
        else:
            simulation["checks_failed"] = issues

        stage_timings["5_output"] = ms(ts)

        latency = round((time.time() - t0) * 1000, 1)

        # ── Build result in shape www/index.html expects ──────────────────────
        result = {
            "success":           simulation.get("can_execute", True) and not issues,
            "request_id":        request_id,
            "intent":            intent,
            "design":            design,
            "schemas": {
                "db":   schemas.get("db",   {}),
                "api":  schemas.get("api",  {}),
                "ui":   schemas.get("ui",   {}),
                "auth": schemas.get("auth", {}),
            },
            "validation":        {"valid": not issues, "errors": flat_errors},
            "simulation_result": simulation,
            "issues_found":      issues,
            "refinement_notes":  notes,
            "assumptions":       intent.get("assumptions", []) + intent.get("ambiguities", []),
            "latency_ms":        latency,
            "metrics": {
                "latency_ms":       latency,
                "stage_timings_ms": stage_timings,
                "repairs":          getattr(pipeline_llm.validator, "repair_count", 0),
            },
        }

        tracker._result = result
        logger.info(f"[{request_id}] Done: success={result['success']} latency={latency}ms")
        await tracker.emit("5_output", "completed", "Ready!")

    except Exception as exc:
        logger.exception(f"[{request_id}] Pipeline crashed: {exc}")
        tracker._result = {
            "success":    False,
            "request_id": request_id,
            "error":      str(exc),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
        tracker.is_complete = True
        await tracker._queue.put({"type": "close"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
