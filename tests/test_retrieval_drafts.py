"""Similar-ticket retrieval + grounded drafting (stub embedder, fake LLM)."""

from supportpilot.drafts.reply import KB_ARTICLES, draft_reply, write_kb
from supportpilot.llm.base import FakeProvider
from supportpilot.retrieval.similar import similar_tickets
from supportpilot.settings import get_config, resolve_path


def test_similar_tickets_topical(indexed):
    hits = similar_tickets("I was charged twice and need a refund")
    assert hits
    assert any(h["category"] == "billing" for h in hits[:3])


def test_similar_respects_top_k(indexed):
    assert len(similar_tickets("slow dashboard", top_k=3)) == 3


def test_kb_written_and_draft_grounded(tmp_path):
    cfg = get_config()
    original = cfg["kb"]["articles_dir"]
    cfg["kb"]["articles_dir"] = str(tmp_path / "kb")
    try:
        write_kb()
        assert len(list(resolve_path(cfg["kb"]["articles_dir"]).glob("*.md"))) == len(KB_ARTICLES)
        provider = FakeProvider(canned="Please use Forgot password [password-reset].")
        result = draft_reply(
            "I cannot log in, password reset email never arrives",
            category="account_access",
            priority="P2",
            similar=[{"category": "account_access", "text": "login broken"}],
            provider=provider,
        )
        assert result["kb_used"]
        prompt = provider.calls[0]["prompt"]
        assert "password-reset" in prompt
        assert "category: account_access" in prompt
    finally:
        cfg["kb"]["articles_dir"] = original


def test_draft_without_kb_flags_escalation(tmp_path):
    cfg = get_config()
    original = cfg["kb"]["articles_dir"]
    cfg["kb"]["articles_dir"] = str(tmp_path / "empty-kb")
    resolve_path(cfg["kb"]["articles_dir"]).mkdir(parents=True)
    try:
        provider = FakeProvider(canned="Escalating.")
        result = draft_reply("mystery issue xyz", "bug", "P3", [], provider=provider)
        assert not result["kb_used"]
    finally:
        cfg["kb"]["articles_dir"] = original
