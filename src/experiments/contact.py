"""The User-Agent the data collectors send. Sites ask scrapers to say who runs them, so it carries a contact address
read from CONTACT_EMAIL in .env: whoever runs a collector is the one sites should reach."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = "bottle-vision-hackathon/0.1"


def user_agent() -> str:
    load_dotenv(ROOT / ".env")
    email = os.environ.get("CONTACT_EMAIL", "").strip()
    if not email:
        sys.exit("CONTACT_EMAIL is not set. Add CONTACT_EMAIL=you@example.com to .env: the collectors put it in "
                 "their User-Agent so site owners can reach whoever runs them.")
    return f"{PRODUCT} ({email})"
