"""
Verification script for the Database Connection Failure path in ConversationGuardian Streamlit App.

Starts a temporary Streamlit instance pointing to an unreachable MongoDB port (27099),
uses Playwright to inspect the UI, and verifies that:
1. The header displays DISCONNECTED.
2. The error banner warns: 'Cannot connect to MongoDB'.
3. Troubleshooting instructions are rendered.
"""

import os
import subprocess
import time
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

def verify_failure_ui():
    env = os.environ.copy()
    env["MONGODB_URI"] = "mongodb://127.0.0.1:27099"  # Deliberately unreachable port
    env["MONGODB_HOST_PORT"] = "27099"

    print("\n--- Testing Database Connection Failure Handling ---")
    print("Launching Streamlit on port 8502 pointing to unreachable endpoint mongodb://127.0.0.1:27099...")
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502", "--server.headless", "true"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        time.sleep(5)  # Wait for startup
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page()
            page.goto("http://localhost:8502", wait_until="networkidle")
            time.sleep(4)

            header_text = page.locator(".cg-header").inner_text()
            assert "DISCONNECTED" in header_text, f"Expected DISCONNECTED in header, got: {header_text}"
            print("  [PASS] Header correctly displays DISCONNECTED status indicator.")

            alert_locator = page.wait_for_selector("text=Cannot connect to MongoDB", timeout=15000)
            assert alert_locator is not None
            page_content = page.content()
            assert "Cannot connect to MongoDB at `mongodb://127.0.0.1:27099`" in page_content or "Cannot connect to MongoDB at" in page_content
            assert "Troubleshooting Instructions" in page_content
            print("  [PASS] UI error banner renders connection failure details and troubleshooting instructions.")

            browser.close()
            print("[PASS] Connection failure path verified successfully!")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    verify_failure_ui()
