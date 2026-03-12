import requests
import subprocess
import time
import os
from datetime import datetime
from ..config import GCLOUD_PATH, GOOGLE_PROJECT_ID, GOOGLE_BUCKET


def veo2(prompt):
    if not GCLOUD_PATH or not GOOGLE_PROJECT_ID or not GOOGLE_BUCKET:
        raise ValueError("GCLOUD_PATH, GOOGLE_PROJECT_ID, and GOOGLE_BUCKET environment variables are required")
    
    token = subprocess.run(
        [GCLOUD_PATH, "auth", "print-access-token"],
        capture_output=True, text=True).stdout.strip()
    base_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{GOOGLE_PROJECT_ID}/locations/us-central1/publishers/google/models/veo-2.0-generate-001"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    resp = requests.post(f"{base_url}:predictLongRunning", headers=headers, json={
        "instances": [{"prompt": prompt}],
        "parameters": {"storageUri": f"gs://{GOOGLE_BUCKET}/veo2_videos/{timestamp}/", "sampleCount": 1, "durationSeconds": 5, "enhancePrompt": False}
    })
    
    if resp.status_code != 200:
        return f"Error: {resp.text}"
    op_id = resp.json()["name"].split("/")[-1]
    
    for i in range(30):
        time.sleep(20)
        status = requests.post(f"{base_url}:fetchPredictOperation", headers=headers, json={
            "operationName": f"projects/{GOOGLE_PROJECT_ID}/locations/us-central1/publishers/google/models/veo-2.0-generate-001/operations/{op_id}"
        }).json()
        
        if status.get("done"):
            if "error" in status:
                return f"Failed: {status['error']}"
            
            gcs_uri = status["response"]["videos"][0]["gcsUri"]
            https_url = gcs_uri.replace("gs://", "https://storage.googleapis.com/")
            
            os.makedirs("veo2_temp", exist_ok=True)
            filename = f"veo2_temp/veo2_{timestamp}.mp4"
            
            with open(filename, 'wb') as f:
                f.write(requests.get(https_url, headers=headers).content)
            
            return filename
    
    return "Timeout"
