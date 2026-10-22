"""Cancellable Hugging Face downloads with bounded retries for transient network failures."""
import json
import sys
import time

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


def download(source):
    for attempt in range(4):
        try:
            snapshot_download(repo_id=source["repo"], allow_patterns=source["patterns"])
            return
        except (httpx.TransportError, HfHubHTTPError, ConnectionError, TimeoutError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if isinstance(error, GatedRepoError) or status in (401, 403) or (status and status < 500 and status not in (408, 429)):
                raise
            if attempt == 3:
                raise
            delay = 2 ** attempt
            print(f"Network error; retrying in {delay}s: {error}", flush=True)
            time.sleep(delay)


def main():
    for source in json.loads(sys.argv[1]):
        print(f"Downloading {source['repo']} {source['patterns']}", flush=True)
        download(source)
    print("done", flush=True)


if __name__ == "__main__":
    main()
