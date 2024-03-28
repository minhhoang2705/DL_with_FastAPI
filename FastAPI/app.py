import sys
from pathlib import Path

import uvicorn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from middleware import LogMiddleware, setup_cors
from routes.base import router

app = FastAPI()

app.add_middleware(LogMiddleware)
setup_cors(app)
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
