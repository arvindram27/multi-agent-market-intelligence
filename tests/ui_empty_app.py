"""Minimal Streamlit app that renders every component with empty/missing state."""
import streamlit as st

from ui.components import competitors, news, tone, topics

news.render([])
topics.render([])
tone.render([])
competitors.render(None, pending=False, on_decision=lambda approved: None)
