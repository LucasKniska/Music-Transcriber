import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import the new function from lilypond.py
from lilypond import convert_to_lilypond 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class Note(BaseModel):
    id: str
    keys: List[str]          
    duration: str            
    rawDuration: Optional[float] = 0.0
    startTimeOffset: Optional[float] = 0.0
    isRest: bool
    color: Optional[str] = "black"

class SessionPayload(BaseModel):
    title: str
    bpm: int
    notes: List[Note]
    createdAt: str

# --- In-Memory Storage ---
current_session: Optional[SessionPayload] = None

# --- API Endpoints ---

@app.post("/api/sessions")
async def save_session(session: SessionPayload):
    global current_session
    print(f"📥 Receiving Session: {session.title}")
    print(f"🎵 BPM: {session.bpm} | Note Count: {len(session.notes)}")
    current_session = session
    return {"message": "Session saved successfully", "count": len(session.notes)}

@app.get("/api/notes")
async def get_latest_notes():
    if current_session:
        return current_session.notes
    return []

@app.delete("/api/notes")
async def clear_session():
    global current_session
    current_session = None
    print("🗑️ Session cleared.")
    return {"message": "Session cleared"}

@app.get("/api/export")
async def export_pdf():
    # Check if we have data to export
    if not current_session or not current_session.notes:
        return Response(content="No notes to export. Please Save Session first.", status_code=400)

    print("📄 Generating PDF...")
    pdf_bytes, error = await convert_to_lilypond(current_session.notes) 
    
    if error:
        return Response(content=error, status_code=500)
        
    return Response(content=pdf_bytes, media_type="application/pdf")

# --- Run Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)