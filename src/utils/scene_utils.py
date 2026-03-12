import re


def extract_scenes(attack_text):
    if attack_text is None or not isinstance(attack_text, str):
        print(f"Warning: extract_scenes received invalid input: {type(attack_text)} - {attack_text}")
        return []
    
    if not attack_text.strip():
        print("Warning: extract_scenes received empty string")
        return []
    
    pattern = r"Scene\s*\d+:\s*(.*?)(?=(?:Scene\s*\d+:)|$)"
    
    try:
        scenes = [scene.strip() for scene in re.findall(pattern, attack_text, re.DOTALL) if scene.strip()]
        return scenes
    except Exception as e:
        print(f"Error in extract_scenes: {e}")
        print(f"Input text: {attack_text[:200]}...")
        return []
