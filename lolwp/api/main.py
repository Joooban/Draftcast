"""FastAPI app: prediction API (and, once built, the React frontend)."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from lolwp.api.predictor import Predictor, ValidationError
from lolwp.api.schemas import PredictRequest, PredictResponse
from lolwp.config import load_settings


def _readable(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"] if p != "body")
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts)


def create_app(artifacts_dir: str | None = None) -> FastAPI:
    artifacts_dir = artifacts_dir or load_settings().artifacts_dir
    app = FastAPI(title="LoL Win Predictor")
    try:
        predictor: Predictor | None = Predictor(artifacts_dir)
    except FileNotFoundError:
        predictor = None

    def ready() -> Predictor:
        if predictor is None:
            raise HTTPException(503, "Models aren't trained yet. Run the build and train steps first.")
        return predictor

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": _readable(exc)})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models_loaded": predictor is not None}

    @app.get("/api/champions")
    def champions():
        return ready().champions()

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            return ready().predict(req)
        except ValidationError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/insights")
    def insights():
        try:
            return ready().insights()
        except FileNotFoundError:
            raise HTTPException(503, "Insights haven't been computed yet.")

    return app


app = create_app()
