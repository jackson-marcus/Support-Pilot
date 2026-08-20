"""Streamlit demo: paste a ticket, get triage + similar tickets + a draft reply."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("SUPPORTPILOT_API_URL", "http://localhost:8090")

st.set_page_config(page_title="supportpilot", page_icon="🎧", layout="wide")
st.title("🎧 supportpilot")
st.caption("Ticket triage, priority, similar-case retrieval, grounded reply drafts")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area(
        "Ticket text",
        "Production is down for our whole team — the dashboard crashes with error 500 when saving a report. Please help urgently.",
        height=120,
    )
with col2:
    plan = st.selectbox("Customer plan", ["basic", "pro", "enterprise"], index=2)
    sentiment = st.slider("Sentiment", -1.0, 1.0, -0.6)
    want_draft = st.checkbox("Draft a reply", value=False)
    provider = st.radio("Provider", ["ollama", "claude", "fake"], horizontal=True)

if st.button("Triage", type="primary") and len(text) >= 10:
    with st.spinner("Triaging…"):
        r = httpx.post(
            f"{API_URL}/triage",
            json={
                "text": text,
                "plan": plan,
                "sentiment": sentiment,
                "draft": want_draft,
                "provider": provider,
            },
            timeout=300,
        )
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
    else:
        body = r.json()
        cls, pri = body["classification"], body["priority"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", cls["category"], delta=f"{cls['confidence']:.0%} confident")
        c2.metric("Priority", pri["band"], delta=f"score {pri['score']}")
        c3.metric("Similar cases", len(body["similar_tickets"]))
        st.caption(
            "Runners-up: "
            + ", ".join(f"{a['category']} ({a['prob']:.0%})" for a in cls["alternatives"])
        )

        st.subheader("Similar resolved tickets")
        st.dataframe(
            pd.DataFrame(body["similar_tickets"]), use_container_width=True, hide_index=True
        )

        if "reply" in body:
            st.subheader(f"Suggested reply ({body['reply']['provider']})")
            st.markdown(body["reply"]["draft"])
            if not body["reply"]["kb_used"]:
                st.warning("No KB article matched — draft recommends escalation.")
