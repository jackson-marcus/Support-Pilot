"""Grounded reply drafting: KB articles + similar tickets -> suggested reply.

The draft is grounded in retrieved KB articles (cited as [article]) and shaped
by triage results; agents review before sending — this assists, never auto-sends.
"""

from __future__ import annotations

from supportpilot.llm.base import LLMProvider
from supportpilot.llm.factory import get_provider
from supportpilot.settings import get_config, resolve_path

SYSTEM = (
    "You draft support replies. Be empathetic, concrete, and brief. Ground every "
    "instruction in the provided KB excerpts, cited as [article-name]. If the KB "
    "does not cover the issue, say the ticket needs escalation instead of inventing steps."
)

PROMPT = """Ticket (category: {category}, priority: {priority}):
{ticket}

Knowledge-base excerpts:
{kb}

Similar resolved tickets:
{similar}

Draft a reply the agent can edit:"""

KB_ARTICLES = {
    "password-reset.md": """# Password reset
Users reset passwords from the login page via "Forgot password". Reset emails
can take up to 5 minutes; check spam. Accounts lock for 15 minutes after 6
failed attempts. Admins can unlock users from Settings > Team > Unlock.""",
    "billing-faq.md": """# Billing FAQ
Duplicate charges are automatically voided within 3 business days; if not,
support can issue a manual refund with the invoice number. Plan changes are
prorated. Receipts are under Settings > Billing > History.""",
    "performance-troubleshooting.md": """# Performance troubleshooting
Slow dashboards are usually large date ranges: suggest 90 days or less, or
scheduled exports. Known degradation windows are posted on the status page.
Collect a HAR file if slowness persists beyond 30 minutes.""",
    "exports.md": """# Data exports
Users export data from Project > Export (CSV or JSON). Exports over 100k rows
run asynchronously and are emailed. Excel export is on the roadmap; CSV opens
in Excel meanwhile.""",
    "team-management.md": """# Team management
Admins add members under Settings > Team > Invite. Roles: viewer, editor,
admin. Invites expire after 7 days. SSO domains auto-join when enabled.""",
}


def write_kb() -> None:
    kb_dir = resolve_path(get_config()["kb"]["articles_dir"])
    kb_dir.mkdir(parents=True, exist_ok=True)
    for name, text in KB_ARTICLES.items():
        (kb_dir / name).write_text(text, encoding="utf-8")


def _tokens(text: str) -> set[str]:
    """Meaningful tokens, lightly normalized (plural 's' stripped)."""
    import re

    raw = re.findall(r"[a-z]+", text.lower())
    return {t.rstrip("s") for t in raw if len(t) > 3}


def _kb_excerpts(ticket_text: str, max_articles: int = 2) -> str:
    """Cheap lexical match over KB articles (few enough to scan fully)."""
    kb_dir = resolve_path(get_config()["kb"]["articles_dir"])
    ticket_words = _tokens(ticket_text)
    scored = []
    for path in sorted(kb_dir.glob("*.md")):
        words = _tokens(path.read_text(encoding="utf-8"))
        scored.append((len(ticket_words & words), path))
    scored.sort(reverse=True, key=lambda t: t[0])
    parts = []
    for overlap, path in scored[:max_articles]:
        if overlap >= 2:
            parts.append(f"--- [{path.stem}] ---\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) or "(no relevant KB article found)"


def draft_reply(
    ticket_text: str,
    category: str,
    priority: str,
    similar: list[dict],
    provider: LLMProvider | None = None,
) -> dict:
    provider = provider or get_provider()
    kb = _kb_excerpts(ticket_text)
    similar_text = (
        "\n".join(f"- ({s['category']}) {s['text'][:140]}" for s in similar[:3]) or "(none)"
    )
    reply = provider.complete(
        PROMPT.format(
            ticket=ticket_text, category=category, priority=priority, kb=kb, similar=similar_text
        ),
        system=SYSTEM,
        max_tokens=get_config()["drafts"]["max_tokens"],
    )
    return {
        "draft": reply,
        "provider": provider.name,
        "kb_used": kb != "(no relevant KB article found)",
    }
