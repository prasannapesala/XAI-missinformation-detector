"""
Run this script in your project folder:
    python patch_app_image.py
"""

import os, sys, shutil
from datetime import datetime

APP_PATH = os.path.join(os.path.dirname(__file__), "src", "dashboard", "app.py")
if not os.path.exists(APP_PATH):
    APP_PATH = os.path.join(os.path.dirname(__file__), "app.py")

if not os.path.exists(APP_PATH):
    print("❌  app.py not found. Run this script from your project root.")
    sys.exit(1)

backup = APP_PATH + f".bak_{datetime.now().strftime('%H%M%S')}"
shutil.copy2(APP_PATH, backup)
print(f"✅  Backup saved → {backup}")

with open(APP_PATH, "r", encoding="utf-8") as f:
    src = f.read()

errors = []

# ══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — Add Image card inside nav-cards-row HTML block
# ══════════════════════════════════════════════════════════════════════════════
OLD1 = '            <div class="card-badge">Auto&nbsp;Fetch &middot; Real-time &middot; PDF</div>\n        </div>\n    </div>\n    """, unsafe_allow_html=True)'

NEW1 = ('            <div class="card-badge">Auto&nbsp;Fetch &middot; Real-time &middot; PDF</div>\n'
        '        </div>\n'
        '        <div class="nav-card">\n'
        '            <span class="card-icon">\U0001f5bc\ufe0f</span>\n'
        '            <div class="card-title">Image Analysis</div>\n'
        '            <div class="card-desc">\n'
        '                Upload an image or paste a URL to detect\n'
        '                deepfakes, extract text (OCR), and verify\n'
        '                the image context for misinformation.\n'
        '            </div>\n'
        '            <div class="card-badge">OCR &middot; Deepfake &middot; Context</div>\n'
        '        </div>\n'
        '    </div>\n'
        '    """, unsafe_allow_html=True)')

if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    print("✅  PATCH 1 applied — Image card added to home page")
else:
    errors.append("PATCH 1 FAILED — Could not find nav-card URL block")
    print("❌  PATCH 1 failed — nav-card URL block not found")

# ══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — Add Image button (columns change + new button)
# ══════════════════════════════════════════════════════════════════════════════
OLD2 = ('    _, col_text, col_url, _ = st.columns([1.2, 1, 1, 1.2])\n'
        '    with col_text:\n'
        '        if st.button("\U0001f4dd  Start Text Analysis", use_container_width=True,\n'
        '                     type="primary", key="nav_text"):\n'
        '            st.session_state.view = "text"\n'
        '            st.rerun()\n'
        '    with col_url:\n'
        '        if st.button("\U0001f517  Analyze a URL", use_container_width=True,\n'
        '                     type="primary", key="nav_url"):\n'
        '            st.session_state.view = "url"\n'
        '            st.rerun()')

NEW2 = ('    _, col_text, col_url, col_img, _ = st.columns([1.2, 1, 1, 1, 1.2])\n'
        '    with col_text:\n'
        '        if st.button("\U0001f4dd  Start Text Analysis", use_container_width=True,\n'
        '                     type="primary", key="nav_text"):\n'
        '            st.session_state.view = "text"\n'
        '            st.rerun()\n'
        '    with col_url:\n'
        '        if st.button("\U0001f517  Analyze a URL", use_container_width=True,\n'
        '                     type="primary", key="nav_url"):\n'
        '            st.session_state.view = "url"\n'
        '            st.rerun()\n'
        '    with col_img:\n'
        '        if st.button("\U0001f5bc\ufe0f  Analyze Image", use_container_width=True,\n'
        '                     type="primary", key="nav_img"):\n'
        '            st.session_state.view = "image"\n'
        '            st.rerun()')

if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    print("✅  PATCH 2 applied — Image button added")
else:
    errors.append("PATCH 2 FAILED — Could not find columns/buttons block")
    print("❌  PATCH 2 failed — buttons block not found")

# ══════════════════════════════════════════════════════════════════════════════
# PATCH 3 — Add image routing after url routing
# ══════════════════════════════════════════════════════════════════════════════
OLD3 = ('if st.session_state.view == "url":\n'
        '    render_url_page()\n'
        '    st.stop()\n'
        '\n'
        '# \u2500\u2500 Keyword lists')

NEW3 = ('if st.session_state.view == "url":\n'
        '    render_url_page()\n'
        '    st.stop()\n'
        '\n'
        '# \u2500\u2500 Image Analysis page \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
        'import IMAGE_TAB_CODE as _itc\n'
        '_itc._hide_sidebar_for_landing = _hide_sidebar_for_landing\n'
        'from IMAGE_TAB_CODE import render_image_page\n'
        '\n'
        'if st.session_state.view == "image":\n'
        '    render_image_page()\n'
        '    st.stop()\n'
        '\n'
        '# \u2500\u2500 Keyword lists')

if OLD3 in src:
    src = src.replace(OLD3, NEW3, 1)
    print("✅  PATCH 3 applied — Image routing added")
else:
    errors.append("PATCH 3 FAILED — Could not find url routing block")
    print("❌  PATCH 3 failed — url routing block not found")

# ── Write patched file ────────────────────────────────────────────────────────
if not errors:
    with open(APP_PATH, "w", encoding="utf-8") as f:
        f.write(src)
    print("\n\U0001f389  All 3 patches applied successfully!")
    print("    Run:  streamlit run src/dashboard/app.py")
else:
    print(f"\n⚠️  {len(errors)} patch(es) failed. Original file NOT modified.")
    print("   Errors:", errors)
    print(f"   Backup still at: {backup}")
