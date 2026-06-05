"""Trello integration — creates one card per real bug found by the debug agent."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
TRELLO_BOARD_NAME = os.getenv("TRELLO_BOARD_NAME", "AI Test Agent")
TRELLO_LIST_NAME = os.getenv("TRELLO_LIST_NAME", "To Do")

BASE_URL = "https://api.trello.com/1"
AUTH = {"key": TRELLO_API_KEY, "token": TRELLO_TOKEN}

# Label color per severity level
SEVERITY_COLORS = {
    "critical": "red",
    "high": "orange",
    "medium": "yellow",
    "low": "blue",
}


def _get_board_id() -> str:
    resp = requests.get(f"{BASE_URL}/members/me/boards", params={**AUTH, "fields": "name,id"})
    resp.raise_for_status()
    boards = resp.json()
    for board in boards:
        if board["name"].lower() == TRELLO_BOARD_NAME.lower():
            return board["id"]
    raise ValueError(
        f"Trello board '{TRELLO_BOARD_NAME}' not found. Available: {[b['name'] for b in boards]}"
    )


def _get_list_id(board_id: str) -> str:
    resp = requests.get(f"{BASE_URL}/boards/{board_id}/lists", params={**AUTH, "fields": "name,id"})
    resp.raise_for_status()
    lists = resp.json()
    for lst in lists:
        if lst["name"].lower() == TRELLO_LIST_NAME.lower():
            return lst["id"]
    raise ValueError(
        f"List '{TRELLO_LIST_NAME}' not found on board. Available: {[l['name'] for l in lists]}"
    )


def _get_existing_cards(list_id: str) -> dict:
    """Return {card_name_lower: card_id} for cards already on the list."""
    resp = requests.get(f"{BASE_URL}/lists/{list_id}/cards", params={**AUTH, "fields": "name"})
    resp.raise_for_status()
    return {card["name"].strip().lower(): card["id"] for card in resp.json()}


def _update_card(card_id: str, desc: str, label_id: str | None) -> dict:
    params = {**AUTH, "desc": desc}
    if label_id:
        params["idLabels"] = label_id
    resp = requests.put(f"{BASE_URL}/cards/{card_id}", params=params)
    resp.raise_for_status()
    return resp.json()


def _get_or_create_label(board_id: str, severity: str) -> str | None:
    color = SEVERITY_COLORS.get(severity.lower())
    if not color:
        return None

    resp = requests.get(f"{BASE_URL}/boards/{board_id}/labels", params={**AUTH, "fields": "name,color,id"})
    resp.raise_for_status()
    for label in resp.json():
        if label.get("name", "").lower() == severity.lower() and label.get("color") == color:
            return label["id"]

    # Create label if it doesn't exist
    resp = requests.post(
        f"{BASE_URL}/labels",
        params={**AUTH, "name": severity.capitalize(), "color": color, "idBoard": board_id},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _format_description(bug: dict, feature_name: str) -> str:
    steps = bug.get("steps_to_reproduce", [])
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) if steps else "N/A"
    return (
        f"**Feature:** {feature_name}\n\n"
        f"**Severity:** {bug.get('severity', 'unknown')}\n\n"
        f"**Steps to Reproduce:**\n{steps_text}\n\n"
        f"**Expected:** {bug.get('expected', 'N/A')}\n\n"
        f"**Actual:** {bug.get('actual', 'N/A')}\n\n"
        f"**Evidence:** {bug.get('evidence', 'N/A')}"
    )


def _create_card(list_id: str, name: str, desc: str, label_id: str | None) -> dict:
    params = {**AUTH, "idList": list_id, "name": name, "desc": desc}
    if label_id:
        params["idLabels"] = label_id
    resp = requests.post(f"{BASE_URL}/cards", params=params)
    resp.raise_for_status()
    return resp.json()


def push_bugs_to_trello(bug_reports: list[dict], feature_name: str = "unknown") -> list[dict]:
    """Create one Trello card per real bug. Skips duplicates. Returns list of created card info."""
    if not bug_reports:
        print("[trello] No bugs to push.")
        return []

    if not TRELLO_API_KEY or not TRELLO_TOKEN:
        print("[trello] TRELLO_API_KEY or TRELLO_TOKEN not set — skipping.")
        return []

    try:
        board_id = _get_board_id()
        list_id = _get_list_id(board_id)
        existing = _get_existing_cards(list_id)
    except Exception as e:
        print(f"[trello] Setup failed: {e}")
        return []

    created = []
    for bug in bug_reports:
        title = bug.get("title", "").strip()
        if not title:
            continue

        card_name = f"[{bug.get('severity', 'bug').upper()}] {title}"
        desc = _format_description(bug, feature_name)
        label_id = _get_or_create_label(board_id, bug.get("severity", ""))

        try:
            if card_name.lower() in existing:
                # Refresh the existing card's description with the latest detail.
                card = _update_card(existing[card_name.lower()], desc, label_id)
                created.append({"name": card_name, "url": card.get("shortUrl", "")})
                print(f"[trello] Card updated: {card_name[:70]} → {card.get('shortUrl', '')}")
            else:
                card = _create_card(list_id, card_name, desc, label_id)
                created.append({"name": card_name, "url": card.get("shortUrl", "")})
                existing[card_name.lower()] = card.get("id", "")
                print(f"[trello] Card created: {card_name[:70]} → {card.get('shortUrl', '')}")
        except Exception as e:
            print(f"[trello] Failed for card '{card_name[:50]}': {e}")

    print(f"[trello] Done — {len(created)} card(s) created.")
    return created
