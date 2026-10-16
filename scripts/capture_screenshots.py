"""
Utility script to capture screenshots of the Streamlit Moderator Dashboard.
Uses Playwright with Microsoft Edge.
"""

from pathlib import Path
import time
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("reports/dashboard_screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def select_nav(page, label_text):
    sidebar = page.locator("[data-testid='stSidebar']")
    sidebar.locator(f"label:has-text('{label_text}')").click()
    time.sleep(3)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1050})
        page = context.new_page()

        print("[1/5] Loading Overview view...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(3)
        page.screenshot(path=str(OUTPUT_DIR / "overview.png"))
        print("  -> Saved overview.png")

        print("[2/5] Navigating to Assess Conversation view...")
        select_nav(page, "Assess Conversation")
        time.sleep(2)
        # Fill in conversation details
        page.locator("input[aria-label='Conversation Identifier']").fill("conv_wikipedia_talk_948")
        page.locator("textarea[aria-label='Turn 1']").fill("I disagree with your removal of the citations on the main article.")
        page.locator("textarea[aria-label='Turn 2']").fill("Your sources fail WP:RS and you keep reverting edits without discussing.")
        page.locator("textarea[aria-label='Turn 3']").fill("Stop being completely incompetent and learn how policies actually work here.")
        # Click Run Risk Assessment
        page.locator("button:has-text('Run Risk Assessment')").click()
        time.sleep(3)
        # Scroll to view the generated risk assessment banner and signals table
        try:
            page.locator(".cg-alert-banner, .cg-nominal-banner").first.scroll_into_view_if_needed()
        except Exception:
            page.evaluate("document.querySelector('section.main').scrollTop = 600")
        time.sleep(1)
        page.screenshot(path=str(OUTPUT_DIR / "assess.png"))
        print("  -> Saved assess.png")

        print("[3/5] Navigating to Assessment History view...")
        select_nav(page, "Assessment History")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(3)
        page.screenshot(path=str(OUTPUT_DIR / "history.png"))
        print("  -> Saved history.png")

        print("[4/5] Navigating to Moderator Review view...")
        select_nav(page, "Moderator Review")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        try:
            # Type an assessment ID into the manual ID input so no dropdown popover stays open
            page.locator("input[aria-label='Or Enter Assessment ID']").fill("asmt_b9870e97bdf8")
            page.keyboard.press("Enter")
            time.sleep(2)
        except Exception as e:
            print(f"  (Manual ID entry fallback: {e})")
        page.screenshot(path=str(OUTPUT_DIR / "review.png"))
        print("  -> Saved review.png")

        print("[5/5] Navigating to Analytics view...")
        select_nav(page, "Analytics")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(3)
        page.screenshot(path=str(OUTPUT_DIR / "analytics.png"))
        print("  -> Saved analytics.png")

        browser.close()
        print("\nAll screenshots captured successfully in reports/dashboard_screenshots/")

if __name__ == "__main__":
    run()
