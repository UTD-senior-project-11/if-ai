import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def central_function():
    return {"Neural": "Nine"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
