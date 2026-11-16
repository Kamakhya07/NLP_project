"""
Instagram Reel Hate-Comment Auto-Moderator.
Uses CyberShield NLP Model API to automatically scan, detect, and hide
bullying or abusive comments on Instagram Reels.

Modes:
  1. Live Mode: Uses Meta / Instagram Graph API (Requires Instagram Professional Account & Access Token)
  2. Simulation Mode: Simulates a live Reel stream with incoming comments to demonstrate auto-moderation.

Usage:
  # Simulation / Demo:
  python scripts/instagram_reel_moderator.py --simulate

  # Live Instagram Reel moderation:
  python scripts/instagram_reel_moderator.py --access-token "YOUR_META_TOKEN" --media-id "REEL_MEDIA_ID"
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import List, Dict, Any
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InstagramModerator")

CYBERSHIELD_API_URL = "http://127.0.0.1:5000/api/predict"
GRAPH_API_BASE = "https://graph.facebook.com/v19.0"

class InstagramReelModerator:
    def __init__(self, api_url: str = CYBERSHIELD_API_URL, access_token: str = None):
        self.api_url = api_url
        self.access_token = access_token

    def classify_comment(self, text: str) -> Dict[str, Any]:
        """Sends text to local CyberShield Flask API for real-time classification."""
        try:
            payload = json.dumps({"text": text}).encode("utf-8")
            req = urllib.request.Request(
                self.api_url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to CyberShield server at {self.api_url}. Is app.py running? Error: {e}")
            return {"status": "error", "should_hide": False, "prediction": "Error"}

    def hide_instagram_comment(self, comment_id: str) -> bool:
        """Calls Instagram Graph API to hide a comment from public view."""
        if not self.access_token:
            logger.warning("[SIMULATION] No access token provided. Simulating comment hide action.")
            return True

        url = f"{GRAPH_API_BASE}/{comment_id}?hide=true&access_token={self.access_token}"
        try:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                return data.get("success", False)
        except Exception as e:
            logger.error(f"Error hiding comment {comment_id} on Instagram: {e}")
            return False

    def moderate_comments(self, comments: List[Dict[str, str]]):
        """Processes a list of Reel comments and hides abusive ones."""
        print("\n" + "=" * 75)
        print(" 🛡️  CYBERSHIELD: INSTAGRAM REEL COMMENT MODERATION REPORT")
        print("=" * 75)
        print(f"{'User':<15} | {'Verdict':<12} | {'Confidence':<10} | {'Action':<15} | Comment")
        print("-" * 75)

        hidden_count = 0
        approved_count = 0

        for item in comments:
            cid = item.get("id", "c000")
            user = item.get("username", "anonymous")
            text = item.get("text", "")

            result = self.classify_comment(text)
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0.0)
            should_hide = result.get("should_hide", False)

            if should_hide:
                action = "🚫 HIDDEN (Hate)"
                hidden_count += 1
                self.hide_instagram_comment(cid)
            else:
                action = "✅ ALLOWED"
                approved_count += 1

            print(f"{user:<15} | {prediction:<12} | {confidence:<9.1%} | {action:<15} | {text}")

        print("=" * 75)
        print(f" Summary: {len(comments)} comments evaluated -> {hidden_count} Hidden, {approved_count} Allowed.")
        print("=" * 75 + "\n")

def run_simulation():
    """Simulates real incoming comments on a newly published Instagram Reel."""
    moderator = InstagramReelModerator()
    simulated_reel_comments = [
        {"id": "101", "username": "priya_art", "text": "Loved the transition at 0:05! Keep posting!"},
        {"id": "102", "username": "troll_x99", "text": "You are completely worthless, stop breathing our air trash"},
        {"id": "103", "username": "fitness_rahul", "text": "Great form and lighting in this reel bro 💪"},
        {"id": "104", "username": "hater_2024", "text": "Nobody asked for your cringe garbage, go die loser"},
        {"id": "105", "username": "sneha_k", "text": "So inspiring, thank you for sharing this! ❤️"},
        {"id": "106", "username": "toxic_user8", "text": "Such an annoying crybaby, delete your account you clown"}
    ]
    moderator.moderate_comments(simulated_reel_comments)

def main():
    parser = argparse.ArgumentParser(description="CyberShield Instagram Reel Comment Moderator")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode with sample Reel comments")
    parser.add_argument("--access-token", type=str, help="Meta / Instagram Graph API Access Token")
    parser.add_argument("--media-id", type=str, help="Instagram Media ID (Reel ID)")
    args = parser.parse_args()

    if args.simulate or not args.access_token:
        logger.info("Starting Instagram Reel comment moderation in SIMULATION mode...")
        run_simulation()
    else:
        logger.info(f"Connecting to Instagram Graph API for Media ID: {args.media_id}")
        moderator = InstagramReelModerator(access_token=args.access_token)
        # Fetch live comments from Graph API
        url = f"{GRAPH_API_BASE}/{args.media_id}/comments?access_token={args.access_token}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            comments = data.get("data", [])
            moderator.moderate_comments(comments)

if __name__ == "__main__":
    main()
