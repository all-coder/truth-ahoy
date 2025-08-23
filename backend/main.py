from fastapi import FastAPI
from routes.routes import router
import uvicorn

app = FastAPI()
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
