import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv()

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_KEY_ID = os.environ.get("R2_SECRET_KEY_ID")

s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_KEY_ID,
    region_name="eeur"
)

if __name__ == "__main__":
    with open('config.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    for n, item in enumerate(config['videos']):
        print(f"Downloading {item['name']}...\n")
        s3.download_file('traffic-engineering', item['name'], f'data/file_{n}.mp4')
    print("Download completed!\n")
