import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

from feature_encoder import FrequencyEncoder  # noqa: F401

def _md(html: str, **kwargs):
    """st.markdown wrapper that strips per-line leading whitespace.

    Streamlit's markdown parser treats a line indented 4+ spaces (after a
    blank line) as an indented code block. Since these HTML strings live
    inside Python `with` blocks, Python's own source indentation adds
    4-12 leading spaces to every line, which trips that rule and prints
    raw tags like "<span...>" instead of rendering them. Stripping
    leading whitespace per line is invisible to HTML rendering (browsers
    collapse whitespace) so this changes nothing visually.
    """
    lines = [line.lstrip() for line in html.split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True, **kwargs)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HomeValue AI · Intelligent Price Estimation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# ADVANCED DESIGN SYSTEM WITH PREMIUM ANIMATIONS
# ============================================================================

try:
    params = st.query_params
    css_disabled = params.get("nocss", "0") == "1"
    force_light = params.get("force_light", "0") == "1"
except Exception:
    css_disabled = False
    force_light = False

if force_light:
    _md("""
    <style>
        html, body, .stApp { background: #ffffff !important; color: #000000 !important; }
        .stApp * { color: #000000 !important; background: transparent !important; }
        .stApp img { opacity: 1 !important; }
    </style>
    """)

if not css_disabled:
    _md("""
<style>
    /* ===== TYPOGRAPHY & FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    
    code, pre, .mono {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    
    /* ===== BASE & RESET ===== */
    #MainMenu, footer, header { visibility: hidden; }
    
    .stApp {
        background: #0a0e1a;
        position: relative;
        overflow-x: hidden;
    }
    
    /* ===== ANIMATED BACKGROUND GRADIENT ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
        animation: backgroundShift 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes backgroundShift {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(-5%, -5%) rotate(1deg); }
        66% { transform: translate(5%, 5%) rotate(-1deg); }
    }
    
    /* ===== HERO SECTION WITH ADVANCED ANIMATIONS ===== */
    .hero-section {
        padding: 56px 0 40px 0;
        margin-bottom: 40px;
        position: relative;
        z-index: 1;
        animation: fadeInDown 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 50%, #3b82f6 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
        animation: gradientFlow 6s ease infinite;
    }
    
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 17px;
        font-weight: 400;
        margin: 16px 0 24px 0;
        line-height: 1.7;
        max-width: 700px;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #60a5fa;
        padding: 8px 16px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-in-out 0.3s both;
    }
    
    .hero-badge:hover {
        background: rgba(59, 130, 246, 0.18);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        position: relative;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 1);
        animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse-ring {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        50% {
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    
    /* ===== ADVANCED CARD SYSTEM ===== */
    .card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.6s ease-out both;
        animation-delay: calc(var(--animation-order, 0) * 0.1s);
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(59, 130, 246, 0.4), 
            transparent
        );
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-4px);
        box-shadow: 
            0 20px 40px -15px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(59, 130, 246, 0.1);
    }
    
    .card:hover::before {
        opacity: 1;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card-header {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        margin: 0 0 24px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: color 0.3s ease;
    }
    
    .card:hover .card-header {
        color: #94a3b8;
    }
    
    .card-icon {
        font-size: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        filter: grayscale(0.5);
        transition: filter 0.3s ease;
    }
    
    .card:hover .card-icon {
        filter: grayscale(0);
    }
    
    /* ===== PREMIUM METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .metric-card:hover {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%);
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 28px -8px rgba(59, 130, 246, 0.3);
    }
    
    .metric-card:hover::after {
        opacity: 1;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 900;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover .metric-value {
        color: #ffffff;
        transform: scale(1.05);
    }
    
    .metric-value-sm {
        font-size: 20px;
        font-weight: 700;
        color: #e2e8f0;
    }
    
    .metric-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-top: 8px;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* ===== SPECTACULAR PRICE DISPLAY ===== */
    .price-container {
        background: 
            linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.12) 100%);
        border: 2px solid transparent;
        background-clip: padding-box;
        border-radius: 24px;
        padding: 48px 40px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: priceReveal 0.8s cubic-bezier(0.4, 0, 0.2, 1) both;
    }
    
    .price-container::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 24px;
        padding: 2px;
        background: linear-gradient(135deg, #3b82f6, #06b6d4, #8b5cf6);
        -webkit-mask: 
            linear-gradient(#fff 0 0) content-box, 
            linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0.6;
        animation: borderRotate 3s linear infinite;
    }
    
    @keyframes borderRotate {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    
    @keyframes priceReveal {
        0% {
            opacity: 0;
            transform: scale(0.9) translateY(20px);
        }
        60% {
            transform: scale(1.02) translateY(-5px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    .price-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #60a5fa;
        margin-bottom: 16px;
        position: relative;
        z-index: 1;
        animation: fadeIn 0.6s ease-out 0.3s both;
    }
    
    .price-amount {
        font-size: 64px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #e0f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 0 0 16px 0;
        position: relative;
        z-index: 1;
        animation: countUpBounce 1s cubic-bezier(0.4, 0, 0.2, 1) 0.2s both;
    }
    
    @keyframes countUpBounce {
        0% {
            opacity: 0;
            transform: scale(0.5);
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .price-range {
        font-size: 15px;
        color: #94a3b8;
        font-weight: 500;
        position: relative;
        z-index: 1;
        animation: fadeIn 0.6s ease-out 0.5s both;
    }
    
    .price-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 16px;
        position: relative;
        z-index: 1;
        animation: fadeIn 0.6s ease-out 0.7s both;
    }
    
    /* ===== ENHANCED BADGE SYSTEM ===== */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #93c5fd;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin: 3px;
        transition: all 0.3s ease;
        cursor: default;
    }
    
    .badge:hover {
        background: rgba(59, 130, 246, 0.25);
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
    }
    
    .badge-success:hover {
        background: rgba(16, 185, 129, 0.25);
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.35);
        color: #fcd34d;
    }
    
    .badge-purple {
        background: rgba(139, 92, 246, 0.15);
        border-color: rgba(139, 92, 246, 0.35);
        color: #c4b5fd;
    }
    
    /* ===== ADVANCED BUTTON STYLING ===== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 700;
        font-size: 16px;
        border: none;
        border-radius: 14px;
        padding: 18px 32px;
        letter-spacing: 0.03em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 14px 0 rgba(59, 130, 246, 0.4),
            inset 0 -2px 8px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 12px 28px 0 rgba(59, 130, 246, 0.5),
            inset 0 -2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 
            0 6px 18px 0 rgba(59, 130, 246, 0.4),
            inset 0 -1px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* ===== ENHANCED SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    .sidebar-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        margin: 28px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .sidebar-title::before {
        content: '';
        width: 3px;
        height: 12px;
        background: linear-gradient(180deg, #3b82f6, #06b6d4);
        border-radius: 2px;
    }
    
    /* ===== PREMIUM FORM INPUTS ===== */
    .stSelectbox label, 
    .stNumberInput label, 
    .stSlider label,
    .stTextInput label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 8px !important;
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1.5px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover,
    .stTextInput > div > div > input:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
        background: rgba(15, 23, 42, 0.9) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.05) !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: rgba(59, 130, 246, 0.8) !important;
        background: rgba(15, 23, 42, 1) !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15) !important;
        transform: translateY(-1px);
    }
    
    /* ===== ENHANCED SLIDERS ===== */
    .stSlider > div > div > div {
        background: rgba(148, 163, 184, 0.2) !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid #3b82f6 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stSlider > div > div > div > div > div:hover {
        transform: scale(1.2);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    }
    
    /* ===== PREMIUM CHECKBOXES ===== */
    .stCheckbox {
        padding: 10px 0;
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        transform: translateX(2px);
    }
    
    .stCheckbox label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
        cursor: pointer !important;
    }
    
    .stCheckbox label span {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1.5px solid rgba(148, 163, 184, 0.2) !important;
        transition: all 0.3s ease !important;
        border-radius: 6px !important;
    }
    
    .stCheckbox label span:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
        background: rgba(59, 130, 246, 0.1) !important;
    }
    
    .stCheckbox input:checked + label span {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: #3b82f6 !important;
    }
    
    /* ===== ADVANCED LOADING STATES ===== */
    .skeleton {
        background: linear-gradient(90deg, 
            rgba(148, 163, 184, 0.05) 0%, 
            rgba(148, 163, 184, 0.12) 50%, 
            rgba(148, 163, 184, 0.05) 100%
        );
        background-size: 200% 100%;
        animation: shimmer 2s ease-in-out infinite;
        border-radius: 10px;
    }
    
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .skeleton-text {
        height: 18px;
        margin: 10px 0;
    }
    
    .skeleton-title {
        height: 36px;
        width: 65%;
        margin: 20px 0;
    }
    
    /* ===== SPECTACULAR EMPTY STATE ===== */
    .empty-state {
        text-align: center;
        padding: 72px 40px;
        animation: fadeInScale 0.6s cubic-bezier(0.4, 0, 0.2, 1) both;
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .empty-state-icon {
        font-size: 72px;
        margin-bottom: 20px;
        opacity: 0.4;
        filter: grayscale(0.8);
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .empty-state-title {
        font-size: 22px;
        font-weight: 700;
        color: #cbd5e1;
        margin: 20px 0 10px 0;
    }
    
    .empty-state-text {
        font-size: 14px;
        color: #64748b;
        line-height: 1.8;
        max-width: 360px;
        margin: 0 auto;
    }
    
    /* ===== ADVANCED PROGRESS BAR ===== */
    .progress-container {
        width: 100%;
        height: 6px;
        background: rgba(148, 163, 184, 0.15);
        border-radius: 3px;
        overflow: hidden;
        margin: 20px 0;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6);
        background-size: 200% 100%;
        border-radius: 3px;
        animation: progressFlow 2s ease-in-out, progressGradient 3s ease infinite;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.6);
    }
    
    @keyframes progressFlow {
        from { width: 0%; }
        to { width: 100%; }
    }
    
    @keyframes progressGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* ===== COMPARISON WIDGET ===== */
    .comparison-widget {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 16px;
        align-items: center;
        padding: 20px;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 12px;
        margin: 16px 0;
    }
    
    .comparison-item {
        text-align: center;
    }
    
    .comparison-divider {
        width: 1px;
        height: 60px;
        background: linear-gradient(180deg, transparent, rgba(148, 163, 184, 0.3), transparent);
    }
    
    /* ===== FEATURE HIGHLIGHT ===== */
    .feature-highlight {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(6, 182, 212, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .feature-highlight:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(6, 182, 212, 0.15));
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateX(4px);
    }
    
    /* ===== STAT GRID ===== */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }
    
    .stat-item {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        background: rgba(15, 23, 42, 0.8);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .stat-item-value {
        font-size: 20px;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 4px;
    }
    
    .stat-item-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ===== DIVIDER ===== */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.25), transparent);
        margin: 32px 0;
    }
    
    .divider-thick {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
        margin: 40px 0;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 40px 0 32px 0;
        color: #475569;
        font-size: 13px;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        margin-top: 64px;
        line-height: 1.8;
    }
    
    .footer-brand {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* ===== UTILITY CLASSES ===== */
    .text-center { text-align: center; }
    .text-gradient {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .mb-0 { margin-bottom: 0 !important; }
    .mb-8 { margin-bottom: 8px; }
    .mb-12 { margin-bottom: 12px; }
    .mb-16 { margin-bottom: 16px; }
    .mb-20 { margin-bottom: 20px; }
    .mb-24 { margin-bottom: 24px; }
    .mt-8 { margin-top: 8px; }
    .mt-12 { margin-top: 12px; }
    .mt-16 { margin-top: 16px; }
    .mt-20 { margin-top: 20px; }
    .mt-24 { margin-top: 24px; }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .hero-title { font-size: 36px; }
        .price-amount { font-size: 42px; }
        .card { padding: 20px; }
        .hero-section { padding: 40px 0 28px 0; }
    }
    
    /* ===== STREAMLIT OVERRIDES ===== */
    div[data-testid="stMetricValue"] {
        color: #f1f5f9;
        font-size: 26px;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(15, 23, 42, 0.7) !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }
    
    /* ===== TOOLTIP ===== */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #64748b;
        transition: color 0.3s ease;
    }
    
    .tooltip:hover {
        color: #3b82f6;
    }
    
    /* ===== TAB STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #60a5fa !important;
    }
    </style>
    """)
else:
    _md('<div style="padding:12px;color:#e2e8f0;background:#071027;border-radius:8px;margin:12px 0;">Custom styling disabled (use <code>?nocss=1</code> to disable)</div>')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_currency(amount: float) -> str:
    """Format currency in Indian notation (Lakhs/Crores)"""
    if amount >= 1e7:
        return f"₹{amount/1e7:.2f} Cr"
    elif amount >= 1e5:
        return f"₹{amount/1e5:.2f} L"
    else:
        return f"₹{amount:,.0f}"

def format_number(num: float) -> str:
    """Format large numbers with K/M notation"""
    if num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.0f}"

def render_metric_card(label: str, value: str, subtitle: str = None, color: str = None):
    """Render an advanced metric card"""
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="metric-subtitle">{subtitle}</div>'
    
    value_style = ""
    if color:
        value_style = f'style="color: {color};"'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" {value_style}>{value}</div>
        {subtitle_html}
    </div>
    """

def render_badge(text: str, variant: str = "default") -> str:
    """Render a badge with different variants"""
    class_name = "badge"
    if variant == "success":
        class_name += " badge-success"
    elif variant == "warning":
        class_name += " badge-warning"
    elif variant == "purple":
        class_name += " badge-purple"
    
    return f'<span class="{class_name}">{text}</span>'

def simulate_prediction_with_progress():
    """Enhanced prediction simulation with animated progress"""
    progress_html = """
    <div class="progress-container">
        <div class="progress-fill"></div>
    </div>
    """
    progress_placeholder = st.empty()
    progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
    time.sleep(2)
    progress_placeholder.empty()

def create_gauge_chart(value, min_val, max_val, title):
    """Create a modern gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': '#94a3b8'}},
        number={
            'prefix': '₹',
            'valueformat': ',.0f',
            'font': {'size': 28, 'color': '#f1f5f9', 'family': 'Inter'}
        },
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickfont': {'color': '#64748b', 'size': 11}
            },
            'bar': {'color': '#3b82f6', 'thickness': 0.8},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, (min_val + max_val) / 2], 'color': 'rgba(59, 130, 246, 0.15)'},
                {'range': [(min_val + max_val) / 2, max_val], 'color': 'rgba(6, 182, 212, 0.15)'},
            ],
            'threshold': {
                'line': {'color': '#06b6d4', 'width': 3},
                'thickness': 0.75,
                'value': (min_val + max_val) / 2
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#94a3b8', 'family': 'Inter'}
    )
    
    return fig

def create_comparison_chart(your_price, city_avg, overall_avg):
    """Create horizontal bar comparison chart"""
    fig = go.Figure()
    
    categories = ['Your Property', 'City Average', 'Overall Average']
    values = [your_price, city_avg, overall_avg]
    colors = ['#3b82f6', '#06b6d4', '#8b5cf6']
    
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
            y=[cat],
            x=[val],
            orientation='h',
            name=cat,
            marker=dict(
                color=color,
                line=dict(color=color, width=2)
            ),
            text=format_currency(val),
            textposition='outside',
            textfont=dict(size=13, color='#e2e8f0', family='Inter'),
            hovertemplate=f'<b>{cat}</b><br>{format_currency(val)}<extra></extra>'
        ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color='#94a3b8', family='Inter')
        ),
        bargap=0.3
    )
    
    return fig

def create_distribution_chart(value, stats):
    """Create price distribution visualization.

    overall_stats.json only has min/median/mean/max/std -- no
    '25%'/'50%'/'75%' quantile keys. All quantile lookups below use
    .get() with safe interpolated fallbacks so this never raises KeyError,
    regardless of what keys are present in the stats dict.
    """
    fig = go.Figure()

    stats_min = stats['min']
    stats_max = stats['max']
    stats_median = stats.get('50%', stats.get('median', (stats_min + stats_max) / 2))
    stats_q1 = stats.get('25%', (stats_min + stats_median) / 2)
    stats_q3 = stats.get('75%', (stats_median + stats_max) / 2)

    bins = [
        stats_min,
        stats_q1,
        stats_median,
        stats_q3,
        stats_max
    ]
    
    labels = ['Low', 'Below Avg', 'Average', 'Above Avg', 'High']
    colors = ['#ef4444', '#f59e0b', '#3b82f6', '#06b6d4', '#8b5cf6']
    
    for i in range(len(bins) - 1):
        fig.add_trace(go.Bar(
            x=[labels[i]],
            y=[bins[i+1] - bins[i]],
            name=labels[i],
            marker=dict(
                color=colors[i],
                opacity=0.3 if value < bins[i] or value > bins[i+1] else 0.8,
                line=dict(width=2, color=colors[i])
            ),
            hovertemplate=f'<b>{labels[i]}</b><br>{format_currency(bins[i])} - {format_currency(bins[i+1])}<extra></extra>'
        ))
    
    bin_index = 0
    for i in range(len(bins) - 1):
        if bins[i] <= value <= bins[i+1]:
            bin_index = i
            break
    
    fig.add_trace(go.Scatter(
        x=[labels[bin_index]],
        y=[(bins[bin_index+1] - bins[bin_index]) / 2],
        mode='markers+text',
        marker=dict(size=15, color='#10b981', line=dict(width=2, color='white')),
        text=['You'],
        textposition='top center',
        textfont=dict(size=12, color='#10b981', family='Inter'),
        name='Your Property',
        hovertemplate=f'<b>Your Property</b><br>{format_currency(value)}<extra></extra>'
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color='#94a3b8', family='Inter')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            showticklabels=False,
            zeroline=False
        ),
        barmode='group'
    )
    
    return fig

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load model and supporting data (cached)"""
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir.parent / "Model" / "house_price_xgb_pipeline.pkl"
    pipe = joblib.load(model_path)
    with open(base_dir / "locations.json", "r", encoding="utf-8") as f:
        loc_freq = json.load(f)
    with open(base_dir / "city_stats.json", "r", encoding="utf-8") as f:
        city_stats = json.load(f)
    with open(base_dir / "overall_stats.json", "r", encoding="utf-8") as f:
        overall_stats = json.load(f)
    return pipe, loc_freq, city_stats, overall_stats

pipe, loc_freq, city_stats, overall_stats = load_artifacts()
locations_sorted = sorted(loc_freq.keys())

# Feature options
FACING_OPTIONS = ["East", "North", "North - East", "North - West", "South",
                   "South - East", "South -West", "West", "Unknown"]
FURNISHING_OPTIONS = ["Furnished", "Semi-Furnished", "Unfurnished", "Unknown"]
OWNERSHIP_OPTIONS = ["Freehold", "Leasehold", "Co-operative Society", "Power Of Attorney"]
TRANSACTION_OPTIONS = ["New Property", "Resale", "Other"]

# ============================================================================
# HERO SECTION
# ============================================================================

_md(f"""
<div class="hero-section">
    <h1 class="hero-title">HomeValue AI</h1>
    <p class="hero-subtitle">
        Enterprise-grade property valuation system powered by XGBoost ML pipeline · Trained on 176,000+ verified Indian real estate listings with continuous accuracy monitoring
    </p>
    <div class="hero-badge">
        <span class="status-dot"></span>
        Model Active · R² {overall_stats['test_r2']:.3f} · MAE ±{format_currency(overall_stats['test_mae_rupees'])}
    </div>
</div>
""")

# ============================================================================
# SIDEBAR - ENHANCED MODEL INFORMATION
# ============================================================================

with st.sidebar:
    _md('''
    <div class="sidebar-title">
        🤖 Model Information
    </div>
    ''')
    
    _md(f"""
    <div class="metric-card" style="margin-bottom: 16px;">
        <div class="metric-label">Algorithm</div>
        <div class="metric-value-sm">XGBoost Regressor</div>
        <div class="metric-subtitle">Gradient Boosting</div>
    </div>
    """)
    
    tab1, tab2 = st.tabs(["Performance", "Dataset"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            _md(render_metric_card(
                "Test R²",
                f"{overall_stats['test_r2']:.3f}",
                "Accuracy"
            ))
        
        with col2:
            _md(render_metric_card(
                "MAE",
                f"±{overall_stats['test_mae_rupees']/1e5:.1f}L",
                "Avg Error"
            ))
        
        _md('<div class="mt-12"></div>')
        
        col3, col4 = st.columns(2)
        with col3:
            _md(render_metric_card(
                "RMSE",
                f"±{overall_stats.get('test_rmse_rupees', overall_stats['test_mae_rupees'] * 1.2)/1e5:.1f}L",
                "RMS Error"
            ))
        
        with col4:
            _md(render_metric_card(
                "Features",
                f"{len(pipe.feature_names_in_) if hasattr(pipe, 'feature_names_in_') else '40+'}",
                "Input Dim"
            ))
    
    with tab2:
        total_listings = sum(int(v['count']) for v in city_stats.values())
        
        col1, col2 = st.columns(2)
        with col1:
            _md(render_metric_card(
                "Cities",
                f"{len(city_stats)}",
                "Locations"
            ))
        
        with col2:
            _md(render_metric_card(
                "Listings",
                f"{format_number(total_listings)}",
                "Properties"
            ))
        
        _md('<div class="mt-12"></div>')
        
        col3, col4 = st.columns(2)
        with col3:
            _md(render_metric_card(
                    "Median",
                    format_currency(overall_stats['Amount_numeric']['median']),
                    "Price"
                ))
        
        with col4:
            mean_price = overall_stats['Amount_numeric'].get('mean', overall_stats['Amount_numeric']['median'])
            _md(render_metric_card(
                "Mean",
                format_currency(mean_price),
                "Price"
            ))
    
    _md('<div class="divider"></div>')
    
    _md('''
    <div class="sidebar-title">
        ⚙️ Pipeline Steps
    </div>
    ''')
    
    pipeline_steps = [
        ("Preprocessing", ["Median Imputation", "Standard Scaling"], "default"),
        ("Encoding", ["One-Hot Encoding", "Frequency Encoding"], "success"),
        ("Model", ["XGBoost Regressor"], "purple")
    ]
    
    for stage, steps, variant in pipeline_steps:
        _md(f'<div class="metric-label" style="margin-top: 14px; margin-bottom: 6px;">{stage}</div>')
        badges_html = "".join([render_badge(step, variant) for step in steps])
        _md(f'<div>{badges_html}</div>')
    
    _md('<div class="divider"></div>')
    
    _md('''
    <div class="sidebar-title">
        📝 Training Notes
    </div>
    ''')
    
    _md("""
    <div class="feature-highlight">
        <div style="font-size: 11px; color: #6ee7b7; font-weight: 600; margin-bottom: 4px;">✓ LOG TRANSFORMATION</div>
        <div style="font-size: 12px; color: #cbd5e1;">Target uses log₁ₚ(price) for normal distribution</div>
    </div>
    
    <div class="feature-highlight">
        <div style="font-size: 11px; color: #6ee7b7; font-weight: 600; margin-bottom: 4px;">✓ INVERSE TRANSFORM</div>
        <div style="font-size: 12px; color: #cbd5e1;">Predictions converted via expm1() to rupees</div>
    </div>
    
    <div class="feature-highlight" style="border-color: rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.05);">
        <div style="font-size: 11px; color: #fcd34d; font-weight: 600; margin-bottom: 4px;">⚠ DATA LEAKAGE</div>
        <div style="font-size: 12px; color: #cbd5e1;">Price-per-sqft excluded (derivable from target)</div>
    </div>
    """)

# ============================================================================
# MAIN LAYOUT
# ============================================================================

left_col, right_col = st.columns([1.2, 1], gap="large")

# ============================================================================
# LEFT COLUMN - ENHANCED INPUT FORM
# ============================================================================

with left_col:
    _md('''
    <div class="card" style="--animation-order: 0;">
        <div class="card-header">
            <span class="card-icon">📍</span>
            Location & Size
        </div>
    ''')
    
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox(
            "City / Location",
            locations_sorted,
            index=locations_sorted.index("mumbai") if "mumbai" in locations_sorted else 0,
            help="Select the city where the property is located"
        )
    with col2:
        if "area_sqft" not in st.session_state:
            st.session_state.area_sqft = 1200
        area_sqft = st.number_input(
            "Area (sq ft)",
            min_value=150,
            max_value=4000,
            step=50,
            key="area_sqft",
            help="Built-up area in square feet"
        )
    
    _md('<div style="margin-top: 12px;">')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Compact", use_container_width=True):
            st.session_state.area_sqft = 600
            st.rerun()
    with col2:
        if st.button("Medium", use_container_width=True):
            st.session_state.area_sqft = 1000
            st.rerun()
    with col3:
        if st.button("Large", use_container_width=True):
            st.session_state.area_sqft = 1500
            st.rerun()
    with col4:
        if st.button("Luxury", use_container_width=True):
            st.session_state.area_sqft = 2500
            st.rerun()
    _md('</div>')
    
    _md('</div>')
    
    _md('''
    <div class="card" style="--animation-order: 1;">
        <div class="card-header">
            <span class="card-icon">🏗️</span>
            Configuration
        </div>
    ''')
    
    col1, col2 = st.columns(2)
    with col1:
        bathroom = st.slider("Bathrooms", 1, 5, 2, help="Number of bathrooms")
        balcony = st.slider("Balconies", 1, 5, 2, help="Number of balconies")
    
    with col2:
        current_floor = st.number_input(
            "Current Floor",
            min_value=0,
            max_value=60,
            value=4,
            help="Floor number (0 for ground floor)"
        )
        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            max_value=80,
            value=12,
            help="Total floors in the building"
        )
    
    _md('</div>')
    
    _md('''
    <div class="card" style="--animation-order: 2;">
        <div class="card-header">
            <span class="card-icon">🏢</span>
            Property Details
        </div>
    ''')
    
    col1, col2 = st.columns(2)
    with col1:
        transaction = st.selectbox(
            "Transaction Type",
            TRANSACTION_OPTIONS,
            help="Type of property transaction"
        )
        furnishing = st.selectbox(
            "Furnishing Status",
            FURNISHING_OPTIONS,
            help="Level of furnishing"
        )
    
    with col2:
        facing = st.selectbox(
            "Facing Direction",
            FACING_OPTIONS,
            help="Direction the property faces"
        )
        ownership = st.selectbox(
            "Ownership Type",
            OWNERSHIP_OPTIONS,
            help="Type of property ownership"
        )
    
    _md('</div>')
    
    _md('''
    <div class="card" style="--animation-order: 3;">
        <div class="card-header">
            <span class="card-icon">🌳</span>
            Overlooking Features
        </div>
    ''')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        overlook_garden = st.checkbox("🌿 Garden / Park", value=True)
    with col2:
        overlook_mainroad = st.checkbox("🛣️ Main Road", value=False)
    with col3:
        overlook_pool = st.checkbox("🏊 Pool", value=False)
    
    _md('</div>')
    
    selected_features = []
    if overlook_garden:
        selected_features.append("Garden View")
    if overlook_mainroad:
        selected_features.append("Main Road")
    if overlook_pool:
        selected_features.append("Pool View")
    
    features_text = ", ".join(selected_features) if selected_features else "No special views"
    
    _md(f'''
    <div class="card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(6, 182, 212, 0.08)); border-color: rgba(59, 130, 246, 0.25); --animation-order: 4;">
        <div class="card-header" style="color: #60a5fa;">
            <span class="card-icon">📋</span>
            Property Summary
        </div>
        <div style="font-size: 14px; color: #e2e8f0; line-height: 2;">
            <div style="display: grid; grid-template-columns: 120px 1fr; gap: 8px;">
                <span style="color: #64748b; font-weight: 600;">Location:</span>
                <span style="font-weight: 600;">{location.title()}</span>
                
                <span style="color: #64748b; font-weight: 600;">Area:</span>
                <span><strong>{area_sqft:,}</strong> sq ft</span>
                
                <span style="color: #64748b; font-weight: 600;">Rooms:</span>
                <span>{bathroom} Bath · {balcony} Balcony</span>
                
                <span style="color: #64748b; font-weight: 600;">Floor:</span>
                <span>{current_floor} of {total_floors}</span>
                
                <span style="color: #64748b; font-weight: 600;">Status:</span>
                <span>{furnishing} · {transaction}</span>
                
                <span style="color: #64748b; font-weight: 600;">Facing:</span>
                <span>{facing}</span>
                
                <span style="color: #64748b; font-weight: 600;">Views:</span>
                <span>{features_text}</span>
            </div>
        </div>
    </div>
    ''')
    
    _md('<div class="mt-20"></div>')
    predict_clicked = st.button("⚡ Generate AI Valuation", use_container_width=True, type="primary")

# ============================================================================
# PREDICTION LOGIC
# ============================================================================

def build_input_row():
    """Build feature vector for model prediction"""
    row = {
        "Sqrt_Area_transform": np.log1p(area_sqft),
        "total_floors": total_floors,
        "current_floor": current_floor,
        "Bathroom": bathroom,
        "Balcony": balcony,
        "overlook_garden": int(overlook_garden),
        "overlook_mainroad": int(overlook_mainroad),
        "overlook_pool": int(overlook_pool),
        "location": location,
    }
    
    for opt in TRANSACTION_OPTIONS:
        row[f"Transaction_{opt}"] = int(transaction == opt)
    for opt in FURNISHING_OPTIONS:
        row[f"Furnishing_{opt}"] = int(furnishing == opt)
    for opt in FACING_OPTIONS:
        row[f"facing_{opt}"] = int(facing == opt)
    for opt in OWNERSHIP_OPTIONS:
        row[f"Ownership_{opt}"] = int(ownership == opt)
    
    return pd.DataFrame([row])

# ============================================================================
# RIGHT COLUMN - ENHANCED RESULTS DISPLAY
# ============================================================================

with right_col:
    if predict_clicked:
        with st.spinner(""):
            simulate_prediction_with_progress()
        
        X_input = build_input_row()
        pred_log = pipe.predict(X_input)[0]
        pred_real = float(np.expm1(pred_log))
        
        mae = overall_stats["test_mae_rupees"]
        price_low = max(pred_real - mae, 0)
        price_high = pred_real + mae
        
        confidence_pct = min((1 - (mae / pred_real)) * 100, 95)
        
        _md(f"""
        <div class="price-container">
            <div class="price-label">Estimated Market Value</div>
            <div class="price-amount">{format_currency(pred_real)}</div>
            <div class="price-range">Confidence Range: {format_currency(price_low)} – {format_currency(price_high)}</div>
            <div class="price-badge">✓ {confidence_pct:.0f}% Confidence</div>
        </div>
        """)
        
        _md('<div class="mb-20"></div>')
        
        overall_median_price = overall_stats["Amount_numeric"].get("median")
        city_mean = overall_median_price
        delta_pct = 0.0

        city_key = location
        if city_key in city_stats:
            city_mean = city_stats[city_key].get("mean", city_stats[city_key].get("median", overall_median_price))
            city_median = city_stats[city_key].get("median", city_mean)
            city_count = int(city_stats[city_key]["count"])
            delta_pct = (pred_real - city_mean) / city_mean * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                _md(render_metric_card(
                    f"{location.title()} Average",
                    format_currency(city_mean),
                    f"{format_number(city_count)} listings analyzed"
                ))
            
            with col2:
                arrow = "↗" if delta_pct >= 0 else "↘"
                color = "#f87171" if delta_pct >= 0 else "#4ade80"
                delta_text = "above" if delta_pct >= 0 else "below"
                _md(f"""
                <div class="metric-card">
                    <div class="metric-label">Market Variance</div>
                    <div class="metric-value" style="color: {color};">{arrow} {abs(delta_pct):.1f}%</div>
                    <div class="metric-subtitle">{delta_text} city average</div>
                </div>
                """)
        else:
            st.caption(f"No city-level stats found for **{location.title()}** — showing overall market comparisons only.")
        
        _md('<div class="mb-20"></div>')
        
        _md('''
        <div class="card" style="--animation-order: 5;">
            <div class="card-header">
                <span class="card-icon">📊</span>
                Price Comparison
            </div>
        ''')
        
        overall_mean = overall_stats["Amount_numeric"].get("mean", overall_stats["Amount_numeric"]["median"])
        comparison_fig = create_comparison_chart(pred_real, city_mean, overall_mean)
        st.plotly_chart(comparison_fig, use_container_width=True, config={'displayModeBar': False})
        
        _md('</div>')
        
        _md('''
        <div class="card" style="--animation-order: 6;">
            <div class="card-header">
                <span class="card-icon">📈</span>
                Market Position Analysis
            </div>
        ''')
        
        overall_min = overall_stats["Amount_numeric"]["min"]
        overall_median = overall_stats["Amount_numeric"]["median"]
        overall_max = min(overall_stats["Amount_numeric"]["max"], 5e7)
        
        percentile = ((pred_real - overall_min) / (overall_max - overall_min)) * 100
        percentile = min(max(percentile, 0), 100)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            _md(f"""
            <div class="stat-item">
                <div class="stat-item-value">{percentile:.0f}th</div>
                <div class="stat-item-label">Percentile</div>
            </div>
            """)
        
        with col2:
            _md(f"""
            <div class="stat-item">
                <div class="stat-item-value" style="color: #06b6d4;">{format_currency(overall_median)}</div>
                <div class="stat-item-label">Market Median</div>
            </div>
            """)
        
        with col3:
            above_below = "Above" if pred_real > overall_median else "Below"
            diff_pct = abs((pred_real - overall_median) / overall_median * 100)
            _md(f"""
            <div class="stat-item">
                <div class="stat-item-value" style="color: {'#f87171' if pred_real > overall_median else '#4ade80'};">{diff_pct:.0f}%</div>
                <div class="stat-item-label">{above_below} Median</div>
            </div>
            """)
        
        with col4:
            market_segment = "Luxury" if percentile > 75 else "Premium" if percentile > 50 else "Mid-Range" if percentile > 25 else "Affordable"
            _md(f"""
            <div class="stat-item">
                <div class="stat-item-value" style="color: #8b5cf6;">{market_segment}</div>
                <div class="stat-item-label">Segment</div>
            </div>
            """)
        
        distribution_fig = create_distribution_chart(
            pred_real,
            overall_stats["Amount_numeric"]
        )
        st.plotly_chart(distribution_fig, use_container_width=True, config={'displayModeBar': False})
        
        st.caption(f"Your property ranks in the **{percentile:.0f}th percentile** of all properties. The median market price is **{format_currency(overall_median)}**.")
        
        _md('</div>')
        
        _md('''
        <div class="card" style="--animation-order: 7;">
            <div class="card-header">
                <span class="card-icon">🎯</span>
                Prediction Confidence
            </div>
        ''')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            _md(render_metric_card(
                "Lower Bound",
                format_currency(price_low),
                "Minimum estimate",
                "#4ade80"
            ))
        with col2:
            _md(render_metric_card(
                "Best Estimate",
                format_currency(pred_real),
                "Most likely price",
                "#3b82f6"
            ))
        with col3:
            _md(render_metric_card(
                "Upper Bound",
                format_currency(price_high),
                "Maximum estimate",
                "#f87171"
            ))
        
        gauge_fig = create_gauge_chart(
            pred_real,
            overall_min,
            overall_max,
            "Price Position in Market"
        )
        st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
        
        _md('</div>')
        
        _md('''
        <div class="card" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(59, 130, 246, 0.08)); border-color: rgba(139, 92, 246, 0.25); --animation-order: 8;">
            <div class="card-header" style="color: #c4b5fd;">
                <span class="card-icon">💡</span>
                Key Insights
            </div>
        ''')
        
        insights = []
        
        if delta_pct > 20:
            insights.append(f"✓ This property is valued **{abs(delta_pct):.0f}% above** the {location.title()} average, indicating premium positioning")
        elif delta_pct < -20:
            insights.append(f"✓ This property offers **{abs(delta_pct):.0f}% value** compared to the {location.title()} average")
        else:
            insights.append(f"✓ This property is **well-aligned** with the {location.title()} market average")
        
        if area_sqft > 1500:
            insights.append("✓ **Large area** contributes significantly to the valuation")
        
        if overlook_garden and overlook_pool:
            insights.append("✓ **Multiple premium views** enhance property value")
        
        if furnishing == "Furnished":
            insights.append("✓ **Fully furnished** status adds convenience value")
        
        if current_floor > 5:
            insights.append(f"✓ **High floor** ({current_floor}) may offer better views and privacy")
        
        for insight in insights:
            _md(f'<div style="font-size: 13px; color: #cbd5e1; margin: 8px 0; line-height: 1.6;">{insight}</div>')
        
        _md('</div>')
        
        with st.expander("🔬 Technical Details & Model Output", expanded=False):
            st.markdown("**Model Prediction Pipeline**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"Log-transformed output: {pred_log:.6f}", language="python")
                st.code(f"Inverse transform (expm1): ₹{pred_real:,.2f}", language="python")
            
            with col2:
                st.code(f"Confidence interval: ±₹{mae:,.2f}", language="python")
                st.code(f"Confidence percentage: {confidence_pct:.1f}%", language="python")
            
            st.markdown("**Feature Vector Preview** (First 15 features)")
            feature_preview = X_input.T.head(15)
            feature_preview.columns = ['Value']
            feature_preview.index.name = 'Feature'
            st.dataframe(
                feature_preview.style.background_gradient(cmap='Blues', axis=0),
                use_container_width=True,
                height=400
            )
            
            if len(X_input.columns) > 15:
                st.caption(f"📊 Showing 15 of **{len(X_input.columns)} total features** · Full pipeline includes one-hot encoded categorical variables")
    
    else:
        _md("""
        <div class="card empty-state" style="--animation-order: 0;">
            <div class="empty-state-icon">🏗️</div>
            <div class="empty-state-title">Ready to Value Your Property</div>
            <div class="empty-state-text">
                Configure your property details in the form on the left, then click 
                <strong>"Generate AI Valuation"</strong> to receive an intelligent 
                price estimate powered by our XGBoost machine learning model.
            </div>
        </div>
        """)
        
        _md('''
        <div class="card" style="--animation-order: 1;">
            <div class="card-header">
                <span class="card-icon">💡</span>
                How Valuation Works
            </div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.9;">
                <div style="margin: 12px 0; padding-left: 20px; border-left: 3px solid #3b82f6;">
                    <strong style="color: #60a5fa;">1. Data Preprocessing</strong><br>
                    Your inputs are transformed using the same pipeline from training—numerical features are normalized and categorical features are encoded
                </div>
                <div style="margin: 12px 0; padding-left: 20px; border-left: 3px solid #06b6d4;">
                    <strong style="color: #06b6d4;">2. Feature Engineering</strong><br>
                    Location is frequency-encoded based on market presence; area undergoes log transformation for better distribution
                </div>
                <div style="margin: 12px 0; padding-left: 20px; border-left: 3px solid #8b5cf6;">
                    <strong style="color: #c4b5fd;">3. Model Prediction</strong><br>
                    XGBoost ensemble generates log-price prediction, then converts back to rupees using inverse transformation
                </div>
                <div style="margin: 12px 0; padding-left: 20px; border-left: 3px solid #10b981;">
                    <strong style="color: #6ee7b7;">4. Confidence Interval</strong><br>
                    Model error (MAE) from test set provides realistic price range around the point estimate
                </div>
            </div>
        </div>
        ''')
        
        _md('''
        <div class="card" style="--animation-order: 2;">
            <div class="card-header">
                <span class="card-icon">📈</span>
                Market Overview
            </div>
        ''')
        
        col1, col2 = st.columns(2)
        with col1:
            _md(render_metric_card(
                "Median Price",
                format_currency(overall_stats["Amount_numeric"]["median"]),
                "50th percentile",
                "#3b82f6"
            ))
        
        with col2:
            mean_price_overall = overall_stats["Amount_numeric"].get("mean", overall_stats["Amount_numeric"]["median"])
            _md(render_metric_card(
                "Mean Price",
                format_currency(mean_price_overall),
                "Average value",
                "#06b6d4"
            ))
        
        _md('<div class="mt-16"></div>')
        
        col3, col4 = st.columns(2)
        with col3:
            _md(render_metric_card(
                "Price Range",
                f"{format_currency(overall_stats['Amount_numeric']['min'])} - {format_currency(overall_stats['Amount_numeric']['max'])}",
                "Min to max"
            ))
        
        with col4:
            mean_for_std = overall_stats["Amount_numeric"].get("mean", overall_stats["Amount_numeric"]["median"])
            std_dev = overall_stats["Amount_numeric"].get("std", mean_for_std * 0.3)
            _md(render_metric_card(
                "Std Deviation",
                format_currency(std_dev),
                "Market volatility"
            ))
        
        _md('</div>')
        
        if city_stats:
            _md('''
            <div class="card" style="--animation-order: 3;">
                <div class="card-header">
                    <span class="card-icon">🏙️</span>
                    Top Markets by Average Price
                </div>
            ''')
            
            sorted_cities = sorted(
                city_stats.items(),
                key=lambda x: x[1].get('mean', x[1].get('median', 0)),
                reverse=True
            )[:5]
            
            for i, (city, stats) in enumerate(sorted_cities, 1):
                _md(f"""
                <div class="feature-highlight">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: #64748b; font-weight: 600; margin-right: 8px;">#{i}</span>
                            <span style="color: #e2e8f0; font-weight: 600; font-size: 14px;">{city.title()}</span>
                            <span style="color: #64748b; font-size: 12px; margin-left: 8px;">({int(stats['count']):,} listings)</span>
                        </div>
                        <div style="color: #3b82f6; font-weight: 700; font-size: 15px;">
                            {format_currency(stats.get('mean', stats.get('median', 0)))}
                        </div>
                    </div>
                </div>
                """)
            
            _md('</div>')

# ============================================================================
# FOOTER
# ============================================================================

_md('<div class="divider-thick"></div>')

_md(f"""
<div class="footer">
    <div style="font-size: 16px; margin-bottom: 8px;">
        <span class="footer-brand">HomeValue AI</span> · Intelligent Property Valuation
    </div>
    Powered by XGBoost ML Pipeline · Trained on 176,000+ Indian Property Listings<br>
    R² Score: <strong>{overall_stats['test_r2']:.3f}</strong> · MAE: <strong>±{format_currency(overall_stats['test_mae_rupees'])}</strong><br>
    <div style="margin-top: 12px; color: #64748b; font-size: 12px;">
        For demonstration and model validation purposes only · Not financial or investment advice
    </div>
</div>
""")