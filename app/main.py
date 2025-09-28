from fastapi import FastAPI, Query
from pipeline import run_pipeline

app = FastAPI(title="Deep-sea eDNA AI Pipeline")

@app.get("/")
def home():
    return {"message": "Deep-sea eDNA AI Pipeline API running"}

@app.post("/analyze")
def analyze_data(url: str = Query(..., description="Dataset URL (FASTA from NCBI/FTP)")):
    """
    Input: URL of dataset (e.g. NCBI FTP link)
    Output: JSON with taxonomy + abundance + clusters
    """
    try:
        result = run_pipeline(url)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
