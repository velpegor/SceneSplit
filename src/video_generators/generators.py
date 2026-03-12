import os
import time
import json
import requests
import subprocess
import jwt
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from lumaai import LumaAI

from ..config import (
    OPENAI_API_KEY, LUMA_API_KEY, KLING_ACCESS_KEY, KLING_SECRET_KEY,
    HAILOU_API_KEY, GCLOUD_PATH, GOOGLE_PROJECT_ID, GOOGLE_BUCKET
)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def initialize_luma_client():
    if not LUMA_API_KEY:
        raise ValueError("LUMA_API_KEY environment variable is required")
    print("Initializing LumaAI client...")
    luma_client = LumaAI(auth_token=LUMA_API_KEY)
    print("LumaAI client initialized")
    return luma_client


def generate_video_with_veo2(prompt, duration_seconds=5, sample_count=1, enhance_prompt=False):
    if not GCLOUD_PATH or not GOOGLE_PROJECT_ID or not GOOGLE_BUCKET:
        raise ValueError("GCLOUD_PATH, GOOGLE_PROJECT_ID, and GOOGLE_BUCKET environment variables are required")
    
    from ..models import veo2
    
    print(f"Generating video with Veo2: {prompt[:100]}...")
    
    try:
        result = veo2.veo2(prompt)
        
        if result and not result.startswith("Error") and not result.startswith("Failed") and result != "Timeout":
            if os.path.exists(result):
                output_dir = "veo2_videos"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(result)
                new_path = os.path.join(output_dir, filename)
                
                import shutil
                shutil.move(result, new_path)
                
                abs_path = os.path.abspath(new_path)
                print(f"Veo2 video generated: {abs_path}")
                return abs_path
            else:
                print(f"Generated file not found: {result}")
                return None
        else:
            print(f"Veo2 generation failed: {result}")
            return None
            
    except Exception as e:
        error_msg = str(e)
        if "'videos'" in error_msg:
            print(f"Veo2 guardrail blocked: {error_msg}")
            return "GUARDRAIL_BLOCKED"
        else:
            print(f"Veo2 generation error: {error_msg}")
            return None


def generate_video_with_sora2(prompt, duration_seconds=4, poll_interval=5, timeout=600):
    if not openai_client:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    try:
        print(f"Generating video with Sora2: {prompt[:100]}...")
        files = {
            "model": (None, "sora-2"),
            "prompt": (None, prompt),
            "seconds": (None, str(duration_seconds)),
            "size": (None, "1280x720")
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        response = requests.post("https://api.openai.com/v1/videos", headers=headers, files=files, timeout=30)
        if response.status_code != 200:
            error_text = response.text
            print(f"Sora2 generation failed: {error_text}")
            lower_text = error_text.lower()
            if response.status_code in (400, 403) and ("policy" in lower_text or "safety" in lower_text or "guardrail" in lower_text):
                return "GUARDRAIL_BLOCKED"
            return None
        result = response.json()
        video_id = result.get("id") or result.get("video_id")
        status = result.get("status")
        if not video_id:
            print(f"Video ID not found in Sora2 response: {result}")
            return None
        start_time = time.time()
        while status not in ("ready", "completed", "succeeded"):
            if status in ("failed", "cancelled"):
                print(f"Sora2 generation failed: {status}")
                return "GUARDRAIL_BLOCKED"
            if time.time() - start_time > timeout:
                print("Sora2 generation timeout")
                return None
            time.sleep(poll_interval)
            status_response = requests.get(f"https://api.openai.com/v1/videos/{video_id}", headers=headers, timeout=30)
            if status_response.status_code != 200:
                print(f"Sora2 status check failed: {status_response.text}")
                lower_status = status_response.text.lower()
                if status_response.status_code in (400, 403) and ("policy" in lower_status or "safety" in lower_status or "guardrail" in lower_status):
                    return "GUARDRAIL_BLOCKED"
                return None
            status_payload = status_response.json()
            status = status_payload.get("status") or status_payload.get("state")
            if status_payload.get("policy_violation"):
                print("Sora2 policy violation detected")
                return "GUARDRAIL_BLOCKED"
        content_response = requests.get(f"https://api.openai.com/v1/videos/{video_id}/content", headers=headers, stream=True, timeout=60)
        if content_response.status_code != 200:
            print(f"Sora2 content download failed: {content_response.text}")
            return None
        output_dir = "sora2_videos"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"sora2_{timestamp}.mp4")
        with open(file_path, "wb") as output_file:
            for chunk in content_response.iter_content(chunk_size=8192):
                if chunk:
                    output_file.write(chunk)
        abs_path = os.path.abspath(file_path)
        print(f"Sora2 video generated: {abs_path}")
        return abs_path
    except requests.RequestException as request_error:
        print(f"Sora2 API request error: {request_error}")
        return None
    except Exception as unexpected_error:
        print(f"Sora2 generation error: {unexpected_error}")
        return None


def generate_video_with_kling(prompt, duration_seconds=5, poll_interval=5, timeout=600):
    if not KLING_ACCESS_KEY or not KLING_SECRET_KEY:
        raise ValueError("KLING_ACCESS_KEY and KLING_SECRET_KEY environment variables are required")
    
    try:
        print(f"Generating video with Kling AI: {prompt[:100]}...")
        
        access_key = KLING_ACCESS_KEY
        secret_key = KLING_SECRET_KEY
        
        base_url = "https://api-singapore.klingai.com"
        api_url = f"{base_url}/v1/videos/text2video"
        
        def encode_jwt_token(ak, sk):
            headers = {
                "alg": "HS256",
                "typ": "JWT"
            }
            payload = {
                "iss": ak,
                "exp": int(time.time()) + 1800,
                "nbf": int(time.time()) - 5
            }
            token = jwt.encode(payload, sk, headers=headers)
            return token
        
        authorization = encode_jwt_token(access_key, secret_key)
        
        headers = {
            "Authorization": f"Bearer {authorization}",
            "Content-Type": "application/json"
        }
        
        duration_str = str(duration_seconds) if duration_seconds in [5, 10] else "5"
        payload = {
            "model_name": "kling-v1",
            "prompt": prompt,
            "duration": duration_str,
            "aspect_ratio": "16:9"
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            error_text = response.text
            print(f"Kling AI generation failed: {error_text}")
            lower_text = error_text.lower()
            if response.status_code in (400, 403) and ("policy" in lower_text or "safety" in lower_text or "guardrail" in lower_text or "moderation" in lower_text or "risk" in lower_text):
                return "GUARDRAIL_BLOCKED"
            return None
        
        result = response.json()
        
        if result.get("code") != 0:
            error_msg = result.get("message", "Unknown error")
            print(f"Kling AI generation failed: {error_msg}")
            lower_msg = error_msg.lower()
            if "policy" in lower_msg or "safety" in lower_msg or "guardrail" in lower_msg or "moderation" in lower_msg or "risk" in lower_msg:
                return "GUARDRAIL_BLOCKED"
            return None
        
        data = result.get("data", {})
        task_id = data.get("task_id")
        
        if not task_id:
            print(f"Task ID not found in Kling AI response: {result}")
            return None
        
        print(f"Kling AI task ID: {task_id}")
        
        status_url = f"{base_url}/v1/videos/text2video/{task_id}"
        start_time = time.time()
        last_print_time = 0
        print_interval = 30
        
        while True:
            if time.time() - start_time > timeout:
                print("Kling AI generation timeout")
                return None
            
            time.sleep(poll_interval)
            
            status_response = requests.get(status_url, headers=headers, timeout=30)
            
            if status_response.status_code != 200:
                print(f"Kling AI status check failed: {status_response.text}")
                lower_status = status_response.text.lower()
                if status_response.status_code in (400, 403) and ("policy" in lower_status or "safety" in lower_status or "guardrail" in lower_status or "risk" in lower_status):
                    return "GUARDRAIL_BLOCKED"
                return None
            
            status_result = status_response.json()
            
            if status_result.get("code") != 0:
                error_msg = status_result.get("message", "Unknown error")
                print(f"Kling AI status check error: {error_msg}")
                return None
            
            status_data = status_result.get("data", {})
            task_status = status_data.get("task_status")
            
            if task_status == "succeed":
                task_result = status_data.get("task_result", {})
                videos = task_result.get("videos", [])
                
                if not videos or len(videos) == 0:
                    print(f"Video info not found in Kling AI response: {status_result}")
                    return None
                
                video_url = videos[0].get("url")
                
                if not video_url:
                    print(f"Video URL not found in Kling AI response: {status_result}")
                    return None
                
                output_dir = "kling_videos"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(output_dir, f"kling_{timestamp}.mp4")
                
                video_response = requests.get(video_url, stream=True, timeout=60)
                if video_response.status_code == 200:
                    with open(file_path, "wb") as output_file:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            if chunk:
                                output_file.write(chunk)
                    abs_path = os.path.abspath(file_path)
                    print(f"Kling AI video generated: {abs_path}")
                    return abs_path
                else:
                    print(f"Kling AI video download failed: HTTP {video_response.status_code}")
                    return None
                
            elif task_status == "failed":
                print(f"Kling AI generation failed: {task_status}")
                task_status_msg = status_data.get("task_status_msg", "")
                if task_status_msg:
                    print(f"Failure reason: {task_status_msg}")
                    lower_msg = task_status_msg.lower()
                    if "policy" in lower_msg or "safety" in lower_msg or "guardrail" in lower_msg or "moderation" in lower_msg or "risk" in lower_msg:
                        return "GUARDRAIL_BLOCKED"
                return None
            
            elif task_status in ("submitted", "processing"):
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    print(f"Kling AI generating... (status: {task_status})")
                    last_print_time = current_time
            else:
                print(f"Kling AI unknown status: {task_status}")
                
    except requests.RequestException as request_error:
        print(f"Kling AI API request error: {request_error}")
        return None
    except Exception as unexpected_error:
        error_msg = str(unexpected_error)
        if "policy" in error_msg.lower() or "safety" in error_msg.lower() or "guardrail" in error_msg.lower() or "moderation" in error_msg.lower() or "risk" in error_msg.lower():
            print(f"Kling AI guardrail blocked: {error_msg}")
            return "GUARDRAIL_BLOCKED"
        print(f"Kling AI generation error: {error_msg}")
        return None


def generate_video_with_luma(prompt, luma_client, model="ray-flash-2", resolution="720p", duration="5s"):
    try:
        print(f"Generating video with Luma: {prompt[:100]}...")
        
        generation = luma_client.generations.create(
            prompt=prompt,
            model=model,
            resolution=resolution,
            duration=duration
        )
        
        print("Luma generating... (waiting for completion)")
        completed = False
        while not completed:
            generation = luma_client.generations.get(id=generation.id)
            if generation.state == "completed":
                completed = True
            elif generation.state == "failed":
                failure_reason = generation.failure_reason
                print(f"Luma generation failed: {failure_reason}")
                if "prompt not allowed" in str(failure_reason):
                    print(f"Luma moderation blocked: {failure_reason}")
                    return "GUARDRAIL_BLOCKED"
                return None
            print("Dreaming")
            time.sleep(10)
        
        output_dir = "luma_videos"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"luma_video_{timestamp}.mp4"
        file_path = os.path.join(output_dir, filename)
        
        video_url = generation.assets.video
        response = requests.get(video_url, stream=True)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            abs_path = os.path.abspath(file_path)
            print(f"Luma video generated: {abs_path}")
            return abs_path
        else:
            print(f"Luma video download failed: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        error_msg = str(e)
        if "prompt not allowed" in error_msg:
            print(f"Luma moderation blocked: {error_msg}")
            return "GUARDRAIL_BLOCKED"
        else:
            print(f"Luma generation error: {error_msg}")
            return None


def invoke_video_generation(model, prompt, api_key):
    print("Submitting video generation task...")
    url = "https://api.minimaxi.chat/v1/video_generation"
    payload = json.dumps({
        "prompt": prompt,
        "model": model
    })
    headers = {
        'authorization': 'Bearer ' + api_key,
        'content-type': 'application/json',
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    task_id = response.json()['task_id']
    print(f"Video generation task submitted successfully, task ID: {task_id}")
    return task_id


def query_video_generation(task_id, api_key):
    url = f"https://api.minimaxi.chat/v1/query/video_generation?task_id={task_id}"
    headers = {
        'authorization': 'Bearer ' + api_key
    }
    response = requests.request("GET", url, headers=headers)
    status = response.json()['status']
    if status == 'Preparing':
        print("...Preparing...")
        return "", 'Preparing'
    elif status == 'Queueing':
        print("...In the queue...")
        return "", 'Queueing'
    elif status == 'Processing':
        print("...Generating...")
        return "", 'Processing'
    elif status == 'Success':
        return response.json()['file_id'], "Finished"
    elif status == 'Fail':
        return "", "Fail"
    else:
        return "", "Unknown"


def fetch_video_result(file_id, api_key, output_file_name):
    print("Video generated successfully, downloading now...")
    url = f"https://api.minimaxi.chat/v1/files/retrieve?file_id={file_id}"
    headers = {
        'authorization': 'Bearer ' + api_key,
    }

    response = requests.request("GET", url, headers=headers)
    print(response.text)

    download_url = response.json()['file']['download_url']
    print(f"Video download link: {download_url}")
    with open(output_file_name, 'wb') as f:
        f.write(requests.get(download_url).content)
    print(f"The video has been downloaded in: {os.getcwd()}/{output_file_name}")


def generate_video_with_hailou(prompt, api_key=None, model="T2V-01-Director"):
    if api_key is None:
        api_key = HAILOU_API_KEY
    if not api_key:
        raise ValueError("HAILOU_API_KEY environment variable is required")
    
    try:
        print(f"Generating video with Hailou: {prompt[:100]}...")
        
        task_id = invoke_video_generation(model, prompt, api_key)
        print("Video generation task submitted")
        
        while True:
            time.sleep(10)
            
            file_id, status = query_video_generation(task_id, api_key)
            if file_id != "":
                output_dir = "hailou_videos"
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hailou_video_{timestamp}.mp4"
                file_path = os.path.join(output_dir, filename)
                
                fetch_video_result(file_id, api_key, file_path)
                
                abs_path = os.path.abspath(file_path)
                print(f"Hailou video generated: {abs_path}")
                return abs_path
            elif status == "Fail":
                print("Guardrail Blocked")
                return "GUARDRAIL_BLOCKED"
            elif status == "Unknown":
                print("Failed")
                return None
                
    except Exception as e:
        error_msg = str(e)
        if "prompt not allowed" in error_msg or "moderation" in error_msg.lower():
            print(f"Hailou moderation blocked: {error_msg}")
            return "GUARDRAIL_BLOCKED"
        else:
            print(f"Hailou generation error: {error_msg}")
            return None
