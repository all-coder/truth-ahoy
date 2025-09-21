from fastapi import FastAPI
from routes.routes import router
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(router, prefix="/api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
