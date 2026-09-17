"""ui/theme.py -- the SatQuery AI design system.

One place for the product's visual rules, so the interface looks like one
application instead of a collection of Streamlit widgets.

The rules are deliberately restrained:

* **one accent** (teal) for anything interactive or selected; everything else is
  neutral slate, because the colour in this product belongs to the data
  (imagery, index ramps, evidence maps), not to the chrome around it;
* **type scale** that gets *smaller* as you go deeper: page title > section >
  subsection > body > caption, so hierarchy is readable at a glance;
* **one surface style** for cards, panels, metrics and tables -- a hairline
  border and a soft shadow, no decorative boxes;
* **interaction states** for hover, focus, active and disabled, because a
  control that does not acknowledge a click does not feel finished.

Nothing here changes a number, a mask or a result: this is presentation only.
"""

from __future__ import annotations

from typing import Final

#: Injected once, at the top of the app.
SATQUERY_CSS: Final[str] = """
<style>
  /* ---------------------------------------------------------- foundations */
  :root {
    --sq-ink:        #0f172a;
    --sq-ink-soft:   #334155;
    --sq-muted:      #64748b;
    --sq-line:       #e2e8f0;
    --sq-line-soft:  #eef2f6;
    --sq-surface:    #ffffff;
    --sq-surface-2:  #f8fafc;
    --sq-accent:     #0e7490;
    --sq-accent-ink: #155e75;
    --sq-accent-bg:  #ecfeff;
    --sq-radius:     10px;
    --sq-shadow:     0 1px 2px rgba(15, 23, 42, .04), 0 1px 3px rgba(15, 23, 42, .06);
  }

  /* Streamlit's own chrome: a product does not ship with a "Made with
     Streamlit" badge, a deploy button or a developer menu. */
  #MainMenu, footer, [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    visibility: hidden; height: 0; position: fixed;
  }
  .viewerBadge_container__1QSob,
  .styles_viewerBadge__1yB5_ { display: none !important; }

  /* ------------------------------------------------------------ typography */
  html, body, [class*="st-"], .stMarkdown, .stText, p, li, label {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
  }
  h1 { font-size: 1.75rem !important; font-weight: 650 !important;
       letter-spacing: -0.02em !important; line-height: 1.2 !important;
       color: var(--sq-ink) !important; margin-bottom: .15rem !important; }
  h2 { font-size: 1.15rem !important; font-weight: 620 !important;
       letter-spacing: -0.01em !important; color: var(--sq-ink) !important;
       padding-top: .5rem !important; }
  h3 { font-size: 1.0rem !important; font-weight: 600 !important;
       color: var(--sq-ink) !important; }
  h4, h5 { font-size: .9rem !important; font-weight: 600 !important;
           color: var(--sq-ink-soft) !important; }
  [data-testid="stCaptionContainer"], .stCaption, small {
    font-size: .8125rem !important; line-height: 1.5 !important;
    color: var(--sq-muted) !important; }
  code, pre, .stCode { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular,
                       Menlo, Consolas, monospace !important; font-size: .8em !important; }

  /* --------------------------------------------------------------- layout */
  .main .block-container { padding-top: 1.25rem; padding-bottom: 3rem;
                           max-width: 1480px; }
  hr, [data-testid="stDivider"] hr { border-color: var(--sq-line) !important; }

  /* ------------------------------------------------------------- surfaces */
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stExpander"],
  [data-testid="stMetric"],
  [data-testid="stDataFrame"],
  [data-testid="stTable"],
  [data-testid="stAlert"] {
    background: var(--sq-surface);
    border: 1px solid var(--sq-line) !important;
    border-radius: var(--sq-radius) !important;
    box-shadow: var(--sq-shadow) !important;
  }
  [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: var(--sq-radius); }

  /* -------------------------------------------------------------- metrics */
  [data-testid="stMetric"] { padding: .7rem .85rem !important; }
  [data-testid="stMetricLabel"] {
    font-size: .75rem !important; font-weight: 600 !important;
    letter-spacing: .01em; color: var(--sq-muted) !important; }
  [data-testid="stMetricValue"] {
    font-size: 1.25rem !important; font-weight: 620 !important;
    letter-spacing: -0.01em; color: var(--sq-ink) !important; }
  [data-testid="stMetricDelta"] { font-size: .75rem !important; }

  /* -------------------------------------------------------------- buttons */
  .stButton > button, .stDownloadButton > button,
  .stFormSubmitContent button, [data-testid="baseButton-secondary"] {
    border-radius: 8px !important; border: 1px solid var(--sq-line) !important;
    background: var(--sq-surface) !important; color: var(--sq-ink-soft) !important;
    font-size: .8125rem !important; font-weight: 550 !important;
    padding: .35rem .8rem !important; transition: background .12s ease,
    border-color .12s ease, color .12s ease, box-shadow .12s ease; }
  .stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--sq-accent) !important; color: var(--sq-accent-ink) !important;
    background: var(--sq-accent-bg) !important; }
  .stButton > button:focus-visible {
    outline: none; box-shadow: 0 0 0 3px rgba(14, 116, 144, .22) !important;
    border-color: var(--sq-accent) !important; }
  .stButton > button:disabled { opacity: .45 !important; cursor: not-allowed; }
  [data-testid="baseButton-primary"], button[kind="primary"] {
    background: var(--sq-accent) !important; border-color: var(--sq-accent) !important;
    color: #fff !important; }
  [data-testid="baseButton-primary"]:hover { background: var(--sq-accent-ink) !important;
    border-color: var(--sq-accent-ink) !important; color: #fff !important; }

  /* --------------------------------------------------------------- inputs */
  .stTextInput input, .stNumberInput input, .stTextArea textarea,
  [data-baseweb="select"] > div, [data-baseweb="input"] > div {
    border-radius: 8px !important; border-color: var(--sq-line) !important;
    background: var(--sq-surface) !important; font-size: .875rem !important; }
  .stTextInput input:focus, .stTextArea textarea:focus,
  [data-baseweb="select"] > div:focus-within {
    border-color: var(--sq-accent) !important;
    box-shadow: 0 0 0 3px rgba(14, 116, 144, .16) !important; }
  [data-testid="stWidgetLabel"] p, .stWidgetLabel p {
    font-size: .8125rem !important; font-weight: 550 !important; color: var(--sq-ink-soft) !important; }

  /* ----------------------------------------------------------- chat turns */
  [data-testid="stChatMessage"] {
    border: 1px solid var(--sq-line); border-radius: var(--sq-radius);
    background: var(--sq-surface); padding: .75rem .9rem; margin-bottom: .6rem;
    box-shadow: var(--sq-shadow); }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--sq-surface-2); border-color: var(--sq-line-soft); }
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"] > div {
    border-radius: 10px !important; border-color: var(--sq-line) !important; }
  [data-testid="stChatInput"]:focus-within > div {
    border-color: var(--sq-accent) !important;
    box-shadow: 0 0 0 3px rgba(14, 116, 144, .16) !important; }

  /* -------------------------------------------------------------- alerts */
  [data-testid="stAlert"] { padding: .7rem .85rem !important; font-size: .875rem; }
  [data-testid="stAlert"] p { font-size: .875rem !important; }

  /* ------------------------------------------------------------ expanders */
  [data-testid="stExpander"] summary {
    font-size: .9rem !important; font-weight: 600 !important;
    color: var(--sq-ink-soft) !important; padding: .55rem .85rem !important; }
  [data-testid="stExpander"] summary:hover { color: var(--sq-accent-ink) !important; }
  [data-testid="stExpander"] > div > div:last-child { padding: 0 .85rem .75rem !important; }

  /* -------------------------------------------------------------- sidebar */
  [data-testid="stSidebar"] {
    border-right: 1px solid var(--sq-line); background: var(--sq-surface-2); }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: var(--sq-ink) !important; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: .8125rem; }

  /* --------------------------------------------------------------- tables */
  [data-testid="stDataFrame"] > div, [data-testid="stTable"] {
    border-radius: var(--sq-radius) !important; overflow: hidden; }
  .stDataFrame th, [data-testid="stTable"] th {
    font-size: .75rem !important; text-transform: uppercase; letter-spacing: .04em;
    color: var(--sq-muted) !important; background: var(--sq-surface-2) !important; }

  /* --------------------------------------------------- product components */
  .sq-header {
    display: flex; align-items: center; gap: .85rem;
    padding: 1rem 1.15rem; margin-bottom: 1rem;
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    border: 1px solid var(--sq-line); border-radius: 12px; box-shadow: var(--sq-shadow); }
  .sq-header img { width: 38px; height: 38px; }
  .sq-header .sq-title { font-size: 1.35rem; font-weight: 660; letter-spacing: -0.02em;
                         color: var(--sq-ink); line-height: 1.15; }
  .sq-header .sq-sub { font-size: .875rem; color: var(--sq-muted); margin-top: .15rem; }
  .sq-header .sq-badge {
    margin-left: auto; font-size: .6875rem; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: var(--sq-accent-ink);
    background: var(--sq-accent-bg); border: 1px solid #cffafe;
    border-radius: 999px; padding: .25rem .6rem; }

  .sq-section { display: flex; align-items: baseline; gap: .6rem;
                margin: 1.5rem 0 .6rem; padding-bottom: .55rem;
                border-bottom: 1px solid var(--sq-line); }
  .sq-section .sq-num { font-size: .6875rem; font-weight: 700; color: var(--sq-accent);
                        background: var(--sq-accent-bg); border-radius: 6px;
                        padding: .15rem .4rem; }
  .sq-section .sq-name { font-size: 1.05rem; font-weight: 620; color: var(--sq-ink);
                         letter-spacing: -0.01em; }
  .sq-section .sq-note { font-size: .8125rem; color: var(--sq-muted); margin-left: auto; }

  .sq-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
  .sq-chip { font-size: .75rem; color: var(--sq-accent-ink); background: var(--sq-accent-bg);
             border: 1px solid #cffafe; border-radius: 999px; padding: .18rem .55rem; }
  .sq-chip-muted { color: var(--sq-muted); background: var(--sq-surface-2);
                   border-color: var(--sq-line); }

  .sq-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
              gap: .7rem; margin: .25rem 0 .75rem; }
  .sq-card { border: 1px solid var(--sq-line); border-radius: var(--sq-radius);
             background: var(--sq-surface); padding: .8rem .9rem; box-shadow: var(--sq-shadow); }
  .sq-card .sq-card-title { font-size: .8125rem; font-weight: 620; color: var(--sq-ink);
                            margin-bottom: .2rem; }
  .sq-card .sq-card-body { font-size: .8125rem; color: var(--sq-muted); line-height: 1.5; }

  .sq-note { font-size: .8125rem; color: var(--sq-muted); line-height: 1.55; }
  .sq-note strong { color: var(--sq-ink-soft); font-weight: 600; }

  @media (max-width: 900px) {
    .sq-cards { grid-template-columns: 1fr 1fr; }
    .sq-header .sq-badge { display: none; }
  }
  @media (max-width: 640px) {
    .sq-cards { grid-template-columns: 1fr; }
    h1 { font-size: 1.4rem !important; }
  }
</style>
"""


#: Small, reusable HTML fragments (kept here so every screen uses the same one).
def section(title: str, note: str = "", number: str = "") -> str:
    """A section header: name, an optional right-aligned note, no rainbow rule."""
    num = f'<span class="sq-num">{number}</span>' if number else ""
    right = f'<span class="sq-note">{note}</span>' if note else ""
    return (f'<div class="sq-section">{num}<span class="sq-name">{title}</span>'
            f'{right}</div>')


def cards(items: "list[tuple[str, str]]") -> str:
    """A responsive row of titled cards."""
    body = "".join(
        f'<div class="sq-card"><div class="sq-card-title">{title}</div>'
        f'<div class="sq-card-body">{text}</div></div>'
        for title, text in items
    )
    return f'<div class="sq-cards">{body}</div>'


def chips(labels: "list[str]", muted: bool = False) -> str:
    """Small pill labels for source, CRS, dates and other metadata."""
    cls = "sq-chip sq-chip-muted" if muted else "sq-chip"
    return ('<div class="sq-chips">'
            + "".join(f'<span class="{cls}">{label}</span>' for label in labels)
            + "</div>")


def note(text: str) -> str:
    """A muted explanatory line (used for limits and caveats)."""
    return f'<div class="sq-note">{text}</div>'


def inject() -> None:
    """Apply the design system. Call once, right after `set_page_config`."""
    import streamlit as st

    st.markdown(SATQUERY_CSS, unsafe_allow_html=True)
