"""
Context-aware CLI chat with Ollama.
Remembers the full conversation history within a session.
Optionally saves/loads history to a JSON file for persistence across sessions.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: run  pip install requests")

# ── Configuration ────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3"          # change to any model you have pulled
HISTORY_FILE  = "chat_history.json"   # set to None to disable persistence
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
# ─────────────────────────────────────────────────────────────────────────────


def load_history(path: str) -> list[dict]:
    """Load previous conversation from disk (if it exists)."""
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Loaded {len(data)} messages from {path}]\n")
        return data
    return []


def save_history(path: str, messages: list[dict]) -> None:
    """Persist conversation to disk."""
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)


def chat(model: str, messages: list[dict]) -> str:
    """
    Send the full message history to Ollama and return the assistant reply.
    Uses streaming so you see tokens as they arrive.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    response.raise_for_status()

    full_reply = []
    print("\nAssistant: ", end="", flush=True)

    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        print(token, end="", flush=True)
        full_reply.append(token)

        if chunk.get("done"):
            break

    print("\n")
    return "".join(full_reply)


def print_help() -> None:
    print(
        "\nSpecial commands:\n"
        "  /quit   or  /exit   — end the session\n"
        "  /clear              — wipe conversation memory\n"
        "  /history            — print all messages so far\n"
        "  /save               — force-save history to disk\n"
        "  /model <name>       — switch model mid-session\n"
        "  /help               — show this message\n"
    )


def main() -> None:
    model = DEFAULT_MODEL

    # Build the initial message list
    messages: list[dict] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # Append persisted history (after the system prompt)
    messages.extend(load_history(HISTORY_FILE))

    print("=" * 55)
    print(f"  Ollama context-aware chat  |  model: {model}")
    print(f"  Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  Type /help for special commands")
    print("=" * 55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Interrupted — saving and exiting]")
            save_history(HISTORY_FILE, _non_system(messages))
            break

        if not user_input:
            continue

        # ── Special commands ─────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()

            if cmd in ("/quit", "/exit"):
                save_history(HISTORY_FILE, _non_system(messages))
                print("[Session saved. Goodbye!]")
                break

            elif cmd == "/clear":
                messages = [m for m in messages if m["role"] == "system"]
                save_history(HISTORY_FILE, [])
                print("[Memory cleared]\n")

            elif cmd == "/history":
                for m in _non_system(messages):
                    role = m["role"].upper()
                    print(f"[{role}] {m['content']}\n")

            elif cmd == "/save":
                save_history(HISTORY_FILE, _non_system(messages))
                print(f"[Saved to {HISTORY_FILE}]\n")

            elif cmd == "/model":
                if len(cmd_parts) > 1:
                    model = cmd_parts[1].strip()
                    print(f"[Switched to model: {model}]\n")
                else:
                    print(f"[Current model: {model}]\n")

            elif cmd == "/help":
                print_help()

            else:
                print(f"[Unknown command: {cmd}  — type /help]\n")

            continue

        # ── Normal message ────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        try:
            reply = chat(model, messages)
        except requests.exceptions.ConnectionError:
            print("[Error: Cannot reach Ollama. Is it running? Try: ollama serve]\n")
            messages.pop()   # remove the user message that failed
            continue
        except requests.exceptions.HTTPError as e:
            print(f"[HTTP error: {e}]\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})

        # Auto-save after every exchange
        save_history(HISTORY_FILE, _non_system(messages))


def _non_system(messages: list[dict]) -> list[dict]:
    """Return only user/assistant messages (skip the system prompt)."""
    return [m for m in messages if m["role"] != "system"]


if __name__ == "__main__":
    main()