import requests
import json
import os

data_folder = "data"

os.makedirs(data_folder, exist_ok=True)

def download_video(name, url):
    file_path = os.path.join(data_folder, f"{name}.mp4")

    if os.path.exists(file_path):
        print(f"Skipping '{file_path}', it already exists.")
        return

    print(f"Downloading '{file_path}' from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete for '{file_path}!")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_path}. Error: {e}")

if __name__ == "__main__":
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found!")
        exit()

    for video in config.get("videos", []):
        download_video(video["name"], video["url"])

    print("\nAll video checks are complete.")