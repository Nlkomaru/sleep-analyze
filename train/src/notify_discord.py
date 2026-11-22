"""Discord webhook notifications for the training pipeline.

stdlib ``urllib`` only — uses a non-default User-Agent because Cloudflare's WAF
rejects the default Python urllib UA with error 1010.

The webhook URL is read from ``SA_DISCORD_WEBHOOK_URL`` or
``DISCORD_WEBHOOK_URL`` via :func:`config.get_env` (``os.environ`` / Colab Secrets
first, then ``train/.env``).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

import config

USER_AGENT = "sleep-analyze-train/1.0"

# Discord embed colors.
COLOR_INFO = 0x5865F2
COLOR_GOOD = 0x57F287
COLOR_WARN = 0xFEE75C
COLOR_BAD = 0xED4245


def webhook_url() -> str | None:
    """Resolve the webhook URL from env / Colab Secrets / ``train/.env``."""
    url = config.get_env("SA_DISCORD_WEBHOOK_URL") or config.get_env(
        "DISCORD_WEBHOOK_URL"
    )
    return url or None


def post(payload: dict) -> bool:
    """Send one webhook payload. Never raises: a lost notification must not kill a run."""
    url = webhook_url()
    if not url:
        print(f"[discord] (no webhook configured) {json.dumps(payload)[:400]}")
        return False
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        print(f"[discord] HTTP {exc.code}: {exc.read()[:400]!r}", file=sys.stderr)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[discord] post failed: {exc}", file=sys.stderr)
    return False


def send_text(text: str) -> bool:
    # Discord hard-caps message content at 2000 characters.
    return post({"content": text[:1990]})


def send_embed(
    *,
    title: str,
    description: str = "",
    fields: list[tuple[str, str, bool]] | None = None,
    color: int = COLOR_INFO,
    footer: str = "",
) -> bool:
    embed: dict = {"title": title[:250], "color": color}
    if description:
        embed["description"] = description[:4000]
    if fields:
        embed["fields"] = [
            {"name": name[:250], "value": (value or "-")[:1020], "inline": inline}
            for name, value, inline in fields[:25]
        ]
    if footer:
        embed["footer"] = {"text": footer[:2040]}
    return post({"embeds": [embed]})


def format_confusion(confusion: list[list[int]], labels: list[str]) -> str:
    """Render a confusion matrix as a fixed-width block (rows = truth)."""
    width = max(len(label) for label in labels)
    header = " " * (width + 3) + " ".join(f"{label[:6]:>6}" for label in labels)
    rows = [header]
    for index, label in enumerate(labels):
        cells = " ".join(f"{value:>6}" for value in confusion[index])
        rows.append(f"{label:<{width}} | {cells}")
    return "```\n" + "\n".join(rows) + "\n```"


def metrics_line(evals: dict[str, dict]) -> str:
    """One-line summary of every eval split's macro-F1."""
    order = ["val", "test_full", "test_hard", "test_legacy"]
    known = [name for name in order if name in evals]
    extra = [name for name in evals if name not in order]
    return " ".join(
        f"{name}={evals[name]['macro_f1']:.4f}" for name in [*known, *extra]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true", help="send one test message"
    )
    parser.add_argument("--text", default="", help="send this text")
    args = parser.parse_args(argv)

    if args.self_test:
        url = webhook_url()
        print(f"[discord] webhook resolved: {'yes' if url else 'NO'}")
        if not url:
            print(
                "[discord] set DISCORD_WEBHOOK_URL (or SA_DISCORD_WEBHOOK_URL)",
                file=sys.stderr,
            )
            return 1
        ok = send_embed(
            title="sleep-analyze train: self-test",
            description="通知経路の確認です。これが見えていれば webhook は生きています。",
            fields=[("source", "notify_discord.py --self-test", True)],
            color=COLOR_GOOD,
        )
        print(f"[discord] delivered: {ok}")
        return 0 if ok else 1

    if args.text:
        return 0 if send_text(args.text) else 1
    parser.error("nothing to do: pass --self-test or --text")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
