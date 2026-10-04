import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gc
import time
import re
import urllib.parse
import tempfile
from datetime import datetime

import requests as _requests
from bs4 import BeautifulSoup
import shap as _shap
import plotly.graph_objects as go
from fpdf import FPDF
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from src.classification.risk_classifier import classify_risk
from realtime_verifier import verify_realtime, blend_verdicts

# ══════════════════════════════════════════════════════════
# URL DETECTOR — helpers ported from app1.py
# ══════════════════════════════════════════════════════════

def _domain_matches(host: str, pattern: str) -> bool:
    host    = host.lower().strip()
    pattern = pattern.lower().strip()
    return host == pattern or host.endswith("." + pattern)

def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev[j-1] + cost)
    return dp[n]

def normalize_leet(s: str) -> str:
    return (s.replace('0','o').replace('1','i').replace('3','e')
             .replace('4','a').replace('5','s').replace('7','t')
             .replace('@','a').replace('$','s'))

TRUSTED_TLD_SUFFIXES = [
    ".gov", ".gov.in", ".gov.uk", ".gov.au", ".gov.us", ".gov.sg",
    ".gov.pk", ".gov.bd", ".gov.np", ".gov.lk", ".gov.ng", ".gov.za",
    ".gov.ke", ".gov.gh", ".gov.nz", ".gov.ca", ".gov.ie", ".gov.ph",
    ".gov.my", ".gov.id", ".gov.br", ".gov.mx", ".gov.ar", ".gov.cl",
    ".nic.in", ".bank.in", ".res.in",
    ".edu", ".edu.in", ".edu.au", ".edu.sg", ".edu.my", ".edu.pk",
    ".edu.bd", ".edu.np", ".edu.ph", ".edu.ng", ".edu.gh", ".edu.ke",
    ".edu.br", ".edu.ar", ".edu.mx", ".edu.co", ".edu.pe",
    ".ac.in", ".ac.uk", ".ac.nz", ".ac.za", ".ac.jp", ".ac.kr",
    ".ac.id", ".ac.th", ".ac.ke", ".ac.ug", ".ac.tz", ".ac.bd",
    ".mil", ".int", ".nato.int",
]

EDUCATIONAL_KEYWORDS = [
    "university","univ","college","institute","iit","iim","nit","iiit",
    "school","academy","edu","vidyalaya","polytechnic","engineering",
    "medical","dental","pharmacy","law","management","technology",
    "research","science","arts","commerce","nursing","highschool",
    "seminary","theological","conservatory","faculty",
]

HOSPITAL_KEYWORDS = [
    "hospital","clinic","health","healthcare","medical","apollo","aiims",
    "fortis","manipal","narayana","care","wellness","pharma","medic",
    "lifecare","medilife","medicity","medanta","nimhans","pgimer",
    "jipmer","sgpgi","tata memorial","cancer","ortho","neuro",
]

GOVT_ORG_KEYWORDS = [
    "govt","government","municipal","corporation","panchayat","nagar",
    "ministry","department","bureau","commission","tribunal","court",
    "police","railway","railways","irctc","defence","military",
    "election","parliament","senate","congress","assembly",
    "secretariat","collectorate","tehsil","taluka","block",
]

CREDIBLE_DOMAINS = [
    "bbc.com","bbc.co.uk","bbc.in","reuters.com","apnews.com","nytimes.com",
    "theguardian.com","washingtonpost.com","cnn.com","npr.org","bloomberg.com",
    "ft.com","economist.com","wsj.com","forbes.com","time.com","newsweek.com",
    "nbcnews.com","cbsnews.com","foxnews.com","abcnews.go.com","aljazeera.com",
    "dw.com","france24.com","euronews.com","theatlantic.com","vice.com","vox.com",
    "axios.com","sky.com","independent.co.uk","telegraph.co.uk","thetimes.co.uk",
    "huffpost.com","businessinsider.com","politico.com","thehill.com",
    "techcrunch.com","wired.com","theverge.com","engadget.com","arstechnica.com",
    "zdnet.com","cnet.com","abc.net.au","smh.com.au","cbc.ca","globeandmail.com",
    "hindustantimes.com","timesofindia.com","ndtv.com","thehindu.com",
    "indianexpress.com","livemint.com","business-standard.com","indiatimes.com",
    "indiatoday.in","news18.com","firstpost.com","scroll.in","thewire.in",
    "deccanherald.com","theprint.in","moneycontrol.com","economictimes.indiatimes.com",
    "financialexpress.com","thequint.com","thenewsminute.com",
    "dawn.com","geo.tv","thenews.com.pk","bdnews24.com","thedailystar.net",
    "bangkokpost.com","thejakartapost.com","rappler.com","inquirer.net",
    "wikipedia.org","britannica.com","snopes.com","factcheck.org",
    "politifact.com","altnews.in","boomlive.in","fullfact.org","leadstories.com",
    "sbi.co.in","hdfcbank.com","icicibank.com","rbi.org.in","sebi.gov.in",
    "google.com","youtube.com","microsoft.com","apple.com","github.com",
    "amazon.com","amazon.in","flipkart.com","ebay.com","walmart.com",
    "facebook.com","instagram.com","twitter.com","x.com","linkedin.com",
    "reddit.com","whatsapp.com","telegram.org",
    "netflix.com","spotify.com","hotstar.com",
    "openai.com","anthropic.com","claude.ai",
    "who.int","cdc.gov","nih.gov","mayoclinic.org","clevelandclinic.org",
    "india.gov.in","mygov.in","uidai.gov.in","irctc.co.in",
    "usa.gov","whitehouse.gov","un.org","worldbank.org",
    "tcs.com","infosys.com","wipro.com","hcltech.com",
]

SUSPICIOUS_TLDS = [
    ".xyz",".top",".click",".biz",".tk",".ml",".ga",
    ".cf",".gq",".pw",".icu",".buzz",".monster",".rest",".loan",
    ".zip",".mov",".cam",".cfd",".cyou",".bond",".hair",".beauty",
    ".skin",".lol",".sbs",
]

CLICKBAIT_PATH_WORDS = [
    "shocking","unbelievable","exposed","banned","conspiracy",
    "leaked","cover-up","mindblowing","you-wont-believe",
    "bombshell","scandalous","secret","hidden","truth-revealed",
    "fake-news","must-see","viral","clickbait","hoax","scam-revealed",
]

KNOWN_BRANDS = [
    "chatgpt","openai","claude","anthropic","gemini","copilot","bard",
    "google","bing","yahoo","duckduckgo","brave","firefox","chrome",
    "facebook","instagram","twitter","tiktok","snapchat","pinterest",
    "reddit","linkedin","youtube","whatsapp","telegram","discord",
    "netflix","spotify","disneyplus","primevideo","hulu","hotstar",
    "flipkart","amazon","ebay","walmart","shopify","etsy","alibaba",
    "aliexpress","meesho","myntra","nykaa","snapdeal",
    "paypal","paytm","phonepe","gpay","razorpay","stripe","visa",
    "mastercard","amex","hdfc","sbi","icici","axisbank","kotak",
    "uber","ola","zomato","swiggy","blinkit","zepto","rapido",
    "microsoft","apple","github","gitlab","dropbox","onedrive",
    "aws","azure","oracle","salesforce","ibm","accenture","infosys",
    "wipro","tcs","cognizant","hcltech","nvidia","intel","amd",
    "airtel","jio","vodafone","bsnl","att","verizon","tmobile",
    "toyota","honda","ford","volkswagen","bmw","mercedes","audi",
    "tesla","hyundai","kia","nissan","maruti","mahindra",
    "pfizer","roche","novartis","merck","bayer","sanofi","astrazeneca",
    "bbc","cnn","nytimes","theguardian","reuters","bloomberg","ndtv",
    "mcdonalds","kfc","pizzahut","dominos","burgerking","subway","starbucks",
    "naukri","linkedin","indeed","glassdoor","monster",
    "magicbricks","99acres","housing","nobroker","zillow",
    "uidai","aadhaar","digilocker","epfo","irctc","indianrailways",
]


def validate_url(url: str) -> tuple:
    url = url.strip()
    if not url:
        return False, "Please enter a URL."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "Could not parse the URL. Please check the format."
    if parsed.scheme not in ("http", "https"):
        return False, "URL must start with http:// or https://"
    host = parsed.hostname or ""
    if not host:
        return False, "URL is missing a valid domain/hostname."
    if "." not in host:
        return False, f"'{host}' is not a valid domain. Did you mean https://{host}.com?"
    if re.search(r"\s", url):
        return False, "URL contains spaces. Please check for typos."
    return True, url


def _detect_typosquat(host: str) -> tuple:
    host = re.sub(r":\d+$", "", host).lower().strip()
    host = re.sub(r"^www\.", "", host)
    if any(_domain_matches(host, d) for d in CREDIBLE_DOMAINS):
        return False, []
    domain_label = re.sub(
        r"\.(com|co\.in|in|net|org|io|info|biz|xyz|top|click|tk|ml|ga|cf|gq|pw|icu|buzz|monster|rest|loan|gov|edu|ac\.in|bank\.in|gov\.in|gov\.uk|edu\.in|co\.uk|com\.au|com\.br|com\.mx)$",
        "", host
    )
    reasons = []
    hit = False
    parts = domain_label.split("-")
    candidates = parts + [domain_label.replace("-", "")]    
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        norm = normalize_leet(candidate.lower())
        for brand in KNOWN_BRANDS:
            dist = levenshtein(norm, brand)
            print("candidate=", candidate, "brand=", brand, "dist=", dist)
            if len(brand) >= 8:   max_dist = 2
            elif len(brand) >= 5: max_dist = 1
            elif len(brand) >= 4: max_dist = 1
            else:                 max_dist = 0
            if (
                dist > 0
                and dist <= max_dist
                and abs(len(norm) - len(brand)) <= 2
                and len(norm) >= 3
            ):
                reasons.append(
                    f"🚩 '{candidate}' resembles '{brand}' (edit distance={dist}, possible typosquat)"
                )
                hit = True
                break
        if hit:
            break
    return hit, reasons


def _extract_url_features(url: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    host   = re.sub(r"^www\.", "", (parsed.hostname or "").lower())
    path   = parsed.path.lower()
    path_q = (path + parsed.query).lower()
    is_trusted_tld    = any(host.endswith(t) for t in TRUSTED_TLD_SUFFIXES)
    is_known_credible = is_trusted_tld or any(_domain_matches(host, d) for d in CREDIBLE_DOMAINS)
    has_ip            = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))
    has_susp_tld      = (not is_trusted_tld) and any(host.endswith(t) for t in SUSPICIOUS_TLDS)
    is_typosquat, _   = _detect_typosquat(host)
    return {
        "domain":              host,
        "scheme":              parsed.scheme,
        "has_https":           parsed.scheme == "https",
        "url_length":          len(url),
        "num_subdomains":      max(0, host.count(".") - 1),
        "is_trusted_tld":      is_trusted_tld,
        "is_known_credible":   is_known_credible,
        "has_suspicious_tld":  has_susp_tld,
        "has_ip_address":      has_ip,
        "has_clickbait_words": any(w in path_q for w in CLICKBAIT_PATH_WORDS),
        "is_typosquat":        is_typosquat,
        "num_digits_in_domain":sum(c.isdigit() for c in host),
        "num_special_chars":   len(re.findall(r"[!@#$%^&*()+=\[\]{}|\\<>]", url)),
        "path_depth":          len([p for p in path.split("/") if p]),
        "has_query_params":    bool(parsed.query),
    }


def _looks_institutional(host: str) -> bool:
    label = re.sub(r"^www\.", "", host.lower())
    label = re.sub(r"\.[a-z.]{2,10}$", "", label).replace("-", "").replace(".", "")
    for kw in EDUCATIONAL_KEYWORDS + HOSPITAL_KEYWORDS + GOVT_ORG_KEYWORDS:
        if kw in label:
            return True
    return False


def _classify_url_type(f: dict) -> str:
    if f["is_typosquat"]:                          return "suspicious"
    if f["is_trusted_tld"]:                        return "institutional"
    if f["is_known_credible"]:                     return "credible"
    if _looks_institutional(f["domain"]):          return "institutional"
    if f["has_suspicious_tld"] or f["has_ip_address"]: return "suspicious"
    return "unknown"


def _score_url_only(features: dict, url: str) -> dict:
    host = features["domain"]
    fake_score = 0.50 if host.endswith(".com") else 0.35
    reasons = []
    if features["is_trusted_tld"]:
        fake_score -= 0.45
        reasons.append("✅ Trusted institutional TLD (.gov/.edu/.bank.in)")
    elif features["is_known_credible"] and not features.get("is_typosquat", False):
        fake_score -= 0.45
        reasons.append("✅ Known credible domain")
    if features["has_https"]:
        fake_score -= 0.04
    else:
        fake_score += 0.10
        reasons.append("🚩 No HTTPS")
    is_typo, typo_reasons = _detect_typosquat(host)
    if is_typo:
        fake_score += 0.45
        reasons.extend(typo_reasons)
    if features["has_suspicious_tld"]:
        fake_score += 0.28
        reasons.append("🚩 Suspicious TLD")
    if features["has_ip_address"]:
        fake_score += 0.35
        reasons.append("🚩 IP address used as domain")
    if features["has_clickbait_words"]:
        fake_score += 0.15
        reasons.append("🚩 Clickbait words in URL path")
    if features["num_digits_in_domain"] >= 2:
        fake_score += 0.08 * min(features["num_digits_in_domain"], 3)
        reasons.append("🚩 Multiple digits in domain")
    fake_score = max(0.05, min(0.95, fake_score))
    real_score = 1.0 - fake_score
    label      = "FAKE" if fake_score >= 0.50 else "REAL"
    return {"fake_score": fake_score, "real_score": real_score,
            "label": label, "confidence": max(fake_score, real_score), "reasons": reasons}


BOT_PAGE_PHRASES = [
    "checking your browser", "verify you're human", "verification failed",
    "please enable javascript", "ddos protection by cloudflare",
    "access denied", "403 forbidden", "robot or human",
    "enable cookies", "just a moment", "captcha", "are you a robot",
    "cloudflare ray id", "security check", "your ip address", "unusual traffic",
]

def _is_bot_page(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in BOT_PAGE_PHRASES)


def _scrape_article(url: str) -> dict:
    strategies = [
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
         "Accept-Language": "en-US,en;q=0.9", "Connection": "keep-alive"},
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
         "Accept-Language": "en-US,en;q=0.5", "Connection": "keep-alive"},
        {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
         "Accept": "text/html"},
    ]
    last_error = None
    for headers in strategies:
        try:
            resp = _requests.get(url, headers=headers, timeout=12, allow_redirects=True)
            resp.raise_for_status()
            soup  = BeautifulSoup(resp.text, "html.parser")
            title = ""
            og = soup.find("meta", property="og:title")
            if og:           title = og.get("content", "").strip()
            elif soup.title: title = soup.title.text.strip()
            desc = ""
            ogd = soup.find("meta", property="og:description")
            if ogd: desc = ogd.get("content", "").strip()
            if not desc:
                md = soup.find("meta", attrs={"name": "description"})
                if md: desc = md.get("content", "").strip()
            for tag in soup(["script","style","nav","footer","header","aside","form","iframe"]):
                tag.decompose()
            article_tag = soup.find("article")
            paragraphs  = article_tag.find_all("p") if article_tag else soup.find_all("p")
            body = " ".join(p.get_text(strip=True) for p in paragraphs[:40])
            full = f"{title}. {desc}. {body}".strip()
            if _is_bot_page(full) or len(body) < 50:
                last_error = "Bot-protection page detected"
                continue
            return {"title": title, "body": body[:2500], "description": desc,
                    "full_text": full[:2500], "error": None}
        except Exception as e:
            last_error = str(e)
            continue
    return {"title":"","body":"","description":"","full_text":"","error": last_error or "Failed to scrape"}


@st.cache_resource(show_spinner=False)
def _load_url_model():
    name = "hamzab/roberta-fake-news-classification"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSequenceClassification.from_pretrained(name)
    mdl.eval()
    return tok, mdl


def _predict_text_url(text: str, tok, mdl) -> dict:
    if not text or not text.strip():
        return {"label":"UNKNOWN","fake_prob":0.5,"real_prob":0.5,"confidence":0.0}
    inp = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = mdl(**inp).logits
    probs    = torch.softmax(logits, dim=1).squeeze().numpy()
    id2label = mdl.config.id2label
    fake_idx = next((k for k,v in id2label.items() if "fake" in v.lower()), 0)
    real_idx = 1 - fake_idx
    fake_p, real_p = float(probs[fake_idx]), float(probs[real_idx])
    return {"label": "FAKE" if fake_p > real_p else "REAL",
            "fake_prob": fake_p, "real_prob": real_p, "confidence": max(fake_p, real_p)}


def _run_lime_url(text: str, tok, mdl, n: int = 10) -> dict:
    if not text or len(text.split()) < 5:
        return {"real":[],"fake":[],"error":"Text too short for LIME analysis"}
    def predict_proba(texts):
        results = []
        for t in texts:
            r = _predict_text_url(t, tok, mdl)
            results.append([r["fake_prob"], r["real_prob"]])
        return np.array(results)
    exp = LimeTextExplainer(class_names=["FAKE","REAL"])
    try:
        e = exp.explain_instance(text, predict_proba, num_features=n, num_samples=200, labels=(0, 1))
        return {"real": e.as_list(label=1), "fake": e.as_list(label=0), "error": None}
    except Exception as ex:
        return {"real":[],"fake":[],"error":f"LIME error: {str(ex)}"}


def _run_shap_url(text: str, tok, mdl, max_tok: int = 80) -> dict:
    if not text or len(text.split()) < 5:
        return {"tokens":[],"fake_shap":[],"real_shap":[],"error":"Text too short for SHAP analysis"}
    def predict_fn(texts):
        results = []
        for t in texts:
            r = _predict_text_url(t, tok, mdl)
            results.append([r["fake_prob"], r["real_prob"]])
        return np.array(results)
    try:
        short  = " ".join(text.split()[:max_tok])
        masker = _shap.maskers.Text(tok)
        expl   = _shap.Explainer(predict_fn, masker, output_names=["FAKE","REAL"])
        sv     = expl([short])
        return {"tokens": list(sv.data[0]), "fake_shap": list(sv.values[0,:,0]),
                "real_shap": list(sv.values[0,:,1]), "error": None}
    except Exception as ex:
        return {"tokens":[],"fake_shap":[],"real_shap":[],"error":f"SHAP error: {str(ex)}"}


_DARK_PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#e6edf3"),
    margin=dict(l=10, r=10, t=40, b=10)
)

def _url_gauge_chart(val, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(val*100, 1),
        number={"suffix":"%","font":{"size":36,"color":color}},
        title={"text":title,"font":{"size":14,"color":"#8b949e"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#30363d"},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"#1c2333","borderwidth":0,
            "steps":[
                {"range":[0,35],"color":"rgba(34,197,94,0.15)"},
                {"range":[35,65],"color":"rgba(245,158,11,0.15)"},
                {"range":[65,100],"color":"rgba(239,68,68,0.15)"}
            ]
        }
    ))
    fig.update_layout(**_DARK_PLOTLY, height=220)
    return fig

def _lime_bar_url(lime_result, label="REAL"):
    data = lime_result.get("real" if label=="REAL" else "fake", [])
    if not data: return None
    words, weights = zip(*data)
    fig = go.Figure(go.Bar(
        x=list(weights), y=list(words), orientation="h",
        marker_color=["#22c55e" if w>0 else "#ef4444" for w in weights],
        text=[f"{w:+.3f}" for w in weights], textposition="outside",
        textfont=dict(size=11, color="#e6edf3")
    ))
    fig.update_layout(**_DARK_PLOTLY,
        title=f"LIME — '{label}' prediction drivers",
        yaxis=dict(autorange="reversed", gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d", zeroline=True, zerolinecolor="#8b949e"),
        height=max(300, len(words)*32+80))
    return fig

def _shap_bar_url(shap_result):
    tokens = shap_result.get("tokens", [])
    vals   = shap_result.get("fake_shap", [])
    if not tokens or not vals: return None
    pairs = sorted(zip(tokens, vals), key=lambda x: abs(x[1]), reverse=True)[:15]
    t, v  = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(v), y=list(t), orientation="h",
        marker_color=["#ef4444" if x>0 else "#22c55e" for x in v],
        text=[f"{x:+.3f}" for x in v], textposition="outside",
        textfont=dict(size=11, color="#e6edf3")
    ))
    fig.update_layout(**_DARK_PLOTLY,
        title="SHAP — token contribution to FAKE probability",
        yaxis=dict(autorange="reversed", gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d", zeroline=True, zerolinecolor="#8b949e"),
        height=max(300, len(t)*32+80))
    return fig

def _url_structure_chart(features):
    checks = {
        "HTTPS Secure":          features["has_https"],
        "Known Credible Domain": features["is_known_credible"] and not features.get("is_typosquat"),
        "Trusted TLD":           features["is_trusted_tld"],
        "No Suspicious TLD":     not features["has_suspicious_tld"],
        "No Clickbait in URL":   not features["has_clickbait_words"],
        "No IP Address":         not features["has_ip_address"],
        "Not Typosquat":         not features.get("is_typosquat", False),
        "Reasonable URL Length": features["url_length"] < 120,
    }
    vals   = [1 if v else 0 for v in checks.values()]
    colors = ["#22c55e" if v else "#ef4444" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=list(checks.keys()), orientation="h",
        marker_color=colors,
        text=["✓ Pass" if v else "✗ Fail" for v in vals],
        textposition="outside", textfont=dict(size=12, color="#e6edf3")
    ))
    fig.update_layout(**_DARK_PLOTLY,
        title="URL Structure Analysis",
        xaxis=dict(range=[0,1.6], visible=False),
        yaxis=dict(gridcolor="#30363d"), height=340)
    return fig


def _safe(text: str) -> str:
    """Strip characters unsupported by FPDF's built-in Latin-1 fonts."""
    return str(text).encode("latin-1", "replace").decode("latin-1")

def _generate_url_pdf(url, result, features, article):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica","B",18)
    pdf.cell(0,12,_safe("XAI Detector - URL Analysis Report"),ln=True)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,6,_safe(f"URL: {url[:80]}"),ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica","B",14)
    if result["label"] == "REAL":
        pdf.set_text_color(34,197,94)
    else:
        pdf.set_text_color(239,68,68)
    pdf.cell(0,10,_safe(f"Verdict: {result['label']}"),ln=True)
    pdf.set_text_color(0,0,0)
    pdf.set_font("Helvetica","",11)
    pdf.cell(0,8,_safe(f"Confidence: {result['confidence']*100:.1f}%  |  Fake Prob: {result['fake_prob']*100:.1f}%"),ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,_safe("URL Features"),ln=True)
    pdf.set_font("Helvetica","",10)
    for k,v in features.items():
        pdf.cell(0,6,_safe(f"  {k}: {v}"),ln=True)
    if article.get("title"):
        pdf.ln(4)
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,_safe("Article"),ln=True)
        pdf.set_font("Helvetica","",10)
        pdf.multi_cell(0,6,_safe(f"Title: {article['title']}"))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="XAI Misinformation Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; font-size: 1.15rem !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #ddeeff !important; font-size: 1.1rem !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #00d4ff !important; font-size: 1.5rem !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #ddeeff !important; font-size: 1.1rem !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #8899aa !important; font-size: 1.1rem !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.5rem !important; }
    [data-testid="stSidebar"] .stToggle label { color: #ddeeff !important; }
    [data-testid="stSidebar"] strong { color: #ffffff !important; }
    [data-testid="stSidebar"] .stAlert { border-radius: 8px; }
    [data-testid="stSidebar"] input[type="password"] {
        background: #0f1a2e !important;
        color: #ddeeff !important;
        border: 1px solid #2d4a6e !important;
        border-radius: 6px !important;
        font-size: 1.1rem !important;
    }
    .stSelectbox > div > div {
        background: #1e2a3a !important;
        color: #ddeeff !important;
        border: 1px solid #2d4a6e !important;
        border-radius: 8px !important;
        font-size: 1.15rem !important;
    }
    .stSelectbox label { color: #8899aa !important; font-size: 1.15rem !important; }
    [data-baseweb="popover"] ul li,
    [data-baseweb="menu"] li,
    [role="option"] { background: #1e2a3a !important; color: #ddeeff !important; font-size: 1.15rem !important; }
    [role="option"]:hover { background: #2d4a6e !important; }
    .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #ddeeff !important; font-size: 1.15rem !important; }
    .stMarkdown ul li { color: #ddeeff !important; font-size: 1.15rem !important; }
    .stMarkdown strong { color: #ffffff !important; }
    .stMarkdown small  { color: #8899aa !important; font-size: 1rem !important; }
    .stTabs [data-baseweb="tab"] { color: #8899aa !important; font-weight: 600; font-size: 1.2rem !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff;
    }
    .stSlider label { color: #ddeeff !important; font-size: 1.15rem !important; }
    .stSlider [data-baseweb="slider"] div { color: #ddeeff !important; }
    [data-testid="stFileUploader"] label { color: #ddeeff !important; font-size: 1.15rem !important; }
    [data-testid="stFileUploader"] section {
        background: #1e2a3a !important;
        border: 1px dashed #2d4a6e !important;
        border-radius: 8px !important;
        color: #ddeeff !important;
        font-size: 1.15rem !important;
    }
    .streamlit-expanderHeader { color: #00d4ff !important; font-size: 1.2rem !important; }
    .streamlit-expanderContent { background: #1e2a3a !important; font-size: 1.15rem !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; font-size: 1.1rem !important; }
    .stTextArea textarea {
        background: #1e2a3a !important;
        color: #e0e0e0 !important;
        border: 1px solid #2d4a6e !important;
        border-radius: 8px !important;
        font-size: 1.15rem !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #00d4ff, #7b2fff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 14px 20px !important;
    }
    .stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #7b2fff) !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    section[data-testid="stSidebar"] { min-width: 320px !important; max-width: 320px !important; transform: none !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #16213e);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 5px;
    }
    .metric-card h2 { color: #00d4ff !important; font-size: 2.5rem; margin: 0; }
    .metric-card p  { color: #8899aa !important; font-size: 1.15rem; margin: 8px 0 0 0; }
    .verdict-fake {
        background: linear-gradient(135deg, #8b0000, #c0392b);
        border-radius: 12px; padding: 24px;
        text-align: center; color: white;
        box-shadow: 0 4px 20px rgba(192,57,43,0.4);
    }
    .verdict-real {
        background: linear-gradient(135deg, #1a5c2a, #27ae60);
        border-radius: 12px; padding: 24px;
        text-align: center; color: white;
        box-shadow: 0 4px 20px rgba(39,174,96,0.4);
    }
    .verdict-fake h1, .verdict-real h1 { font-size: 3rem; margin: 0; color: white !important; }
    .verdict-fake p,  .verdict-real p  { font-size: 1.2rem; margin: 8px 0 0 0; opacity: 0.9; color: white !important; }
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #556677 !important;
        font-size: 1.3rem;
        margin-top: 0;
        letter-spacing: 2px;
    }
    .section-header {
        color: #00d4ff !important;
        font-size: 1.4rem;
        font-weight: 700;
        border-left: 3px solid #00d4ff;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }
    .explain-box {
        background: #1e2a3a;
        border: 1px solid #2d4a6e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #ddeeff !important;
        line-height: 1.6;
        font-size: 1.15rem;
    }
    .keyword-fake {
        background: #7b241c;
        color: white !important;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 1.1rem;
        margin: 4px;
        display: inline-block;
    }
    .keyword-real {
        background: #1a5c2a;
        color: white !important;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 1.1rem;
        margin: 4px;
        display: inline-block;
    }
    .stAlert p { color: inherit !important; font-size: 1.15rem !important; }
    [data-testid="stMetricLabel"] { color: #8899aa !important; font-size: 1.15rem !important; }
    [data-testid="stMetricValue"] { color: #ddeeff !important; font-size: 1.5rem !important; }
    .welcome-title {
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.25rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.25;
    }
    .welcome-sub {
        text-align: center;
        color: #8899aa !important;
        font-size: 1.3rem;
        letter-spacing: 1px;
        margin-top: 12px;
    }
    .nav-cards-row {
        display: flex;
        justify-content: center;
        gap: 28px;
        flex-wrap: wrap;
        padding: 0 24px;
        margin-bottom: 28px;
    }
    .nav-card {
        background: linear-gradient(135deg, #1a2235 0%, #16213e 100%);
        border: 1px solid #2d4a6e;
        border-radius: 20px;
        padding: 38px 32px 30px 32px;
        width: 300px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .nav-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 8px 36px rgba(0, 212, 255, 0.18);
        transform: translateY(-5px);
    }
    .nav-card .card-icon  { font-size: 3.5rem; margin-bottom: 14px; display: block; }
    .nav-card .card-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .nav-card .card-desc  {
        color: #8899aa;
        font-size: 1.1rem;
        line-height: 1.65;
        margin-bottom: 0;
    }
    .nav-card .card-badge {
        display: inline-block;
        margin-top: 18px;
        background: linear-gradient(90deg, #00d4ff18, #7b2fff18);
        border: 1px solid #2d4a6e;
        color: #00d4ff;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 20px;
        padding: 6px 16px;
        letter-spacing: 0.4px;
    }
    .home-btn-row {
        display: flex;
        justify-content: center;
        gap: 24px;
        flex-wrap: wrap;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────
MODEL_NAME = "roberta-base"
MAX_LEN    = 64
SAVED_PATH = "D:\\roberta_saved"

# ── Session state ────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "last_pred"     not in st.session_state: st.session_state.last_pred     = None
if "last_risk"     not in st.session_state: st.session_state.last_risk     = None
if "last_shap"     not in st.session_state: st.session_state.last_shap     = []
if "last_lime"     not in st.session_state: st.session_state.last_lime     = []
if "last_text"     not in st.session_state: st.session_state.last_text     = ""
if "last_fkws"     not in st.session_state: st.session_state.last_fkws     = []
if "last_rkws"     not in st.session_state: st.session_state.last_rkws     = []
if "batch_results" not in st.session_state: st.session_state.batch_results = None
if "view"          not in st.session_state: st.session_state.view          = "home"


def _hide_sidebar_for_landing():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        section[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════
# HOME PAGE  (card layout)
# ════════════════════════════════════════════════════════
def render_home_page():
    _hide_sidebar_for_landing()
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem !important; }
    </style>
    <div style="text-align:center; margin-bottom: 18px;">
        <svg width="90" height="96" viewBox="0 0 90 96" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="outerGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#c8d8e8;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#8aaabb;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="leftPanel" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#55bbee;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#44aadd;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="rightPanel" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#2277bb;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#1a5fa0;stop-opacity:1" />
                </linearGradient>
            </defs>
            <!-- Outer shield (silver/grey border) -->
            <path d="M45 4 L84 18 L84 50 C84 72 66 88 45 94 C24 88 6 72 6 50 L6 18 Z"
                  fill="url(#outerGrad)" />
            <!-- Left blue panel -->
            <path d="M45 13 L45 87 C27 80 14 65 14 50 L14 25 Z"
                  fill="url(#leftPanel)" />
            <!-- Right darker blue panel -->
            <path d="M45 13 L76 25 L76 50 C76 65 63 80 45 87 Z"
                  fill="url(#rightPanel)" />
            <!-- Top highlight on outer shield -->
            <path d="M45 4 L84 18 L84 22 L45 10 L6 22 L6 18 Z"
                  fill="rgba(255,255,255,0.35)" />
            <!-- Left panel shine -->
            <path d="M18 28 L44 18 L44 40 C36 38 24 36 18 32 Z"
                  fill="rgba(255,255,255,0.22)" />
        </svg>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<h1 class="welcome-title">Welcome to Misinformation Detector</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="welcome-sub">Choose how you want to analyze content</p>',
        unsafe_allow_html=True,
    )

    # ── Visual cards (HTML only – no interaction) ─────────
    st.markdown("""
    <div class="nav-cards-row">
        <div class="nav-card">
            <span class="card-icon">📝</span>
            <div class="card-title">Text Analysis</div>
            <div class="card-desc">
                Paste any news headline, claim, or article text
                and get instant fake&nbsp;/&nbsp;real detection
                with full XAI breakdown.
            </div>
            <div class="card-badge">SHAP &middot; LIME &middot; Risk Score</div>
        </div>
        <div class="nav-card">
            <span class="card-icon">🔗</span>
            <div class="card-title">URL Analysis</div>
            <div class="card-desc">
                Enter a news article URL and we'll fetch the
                content automatically, then analyze it for
                misinformation signals.
            </div>
            <div class="card-badge">Auto&nbsp;Fetch &middot; Real-time &middot; PDF</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Streamlit buttons centered below the cards ────────
    _, col_text, col_url, _ = st.columns([1.2, 1, 1, 1.2])
    with col_text:
        if st.button("📝  Start Text Analysis", use_container_width=True,
                     type="primary", key="nav_text"):
            st.session_state.view = "text"
            st.rerun()
    with col_url:
        if st.button("🔗  Analyze a URL", use_container_width=True,
                     type="primary", key="nav_url"):
            st.session_state.view = "url"
            st.rerun()
            # ── About / How it works section ──────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .about-divider { border: none; border-top: 1px solid #2d4a6e; margin: 0 0 2.5rem; }
    .about-badge {
        display: inline-block; font-size: 11px; font-weight: 700;
        padding: 4px 12px; border-radius: 20px; margin-bottom: 10px;
        letter-spacing: 0.5px; text-transform: uppercase;
    }
    .about-h2 { color: #ffffff; font-size: 1.35rem; font-weight: 800; margin: 0 0 6px; }
    .about-sub { color: #8899aa; font-size: 0.88rem; margin: 0 0 1.2rem; line-height: 1.6; }
    .about-step-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 2.5rem;
    }
    .about-step-card {
        background: linear-gradient(135deg, #1a2235, #16213e);
        border: 1px solid #2d4a6e; border-radius: 14px; padding: 1.1rem 1rem;
    }
    .about-step-num { font-size: 11px; font-weight: 700; color: #00d4ff; margin-bottom: 6px; }
    .about-step-title { color: #ffffff; font-size: 0.95rem; font-weight: 700; margin-bottom: 6px; }
    .about-step-desc { color: #8899aa; font-size: 0.82rem; line-height: 1.55; }
    .about-xai-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 1rem; }
    .about-xai-card {
        background: linear-gradient(135deg, #1a2235, #16213e);
        border: 1px solid #2d4a6e; border-radius: 14px; padding: 1.3rem 1.25rem;
    }
    .about-xai-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .about-xai-icon {
        width: 38px; height: 38px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; flex-shrink: 0;
    }
    .about-xai-title { color: #ffffff; font-size: 1rem; font-weight: 800; }
    .about-xai-full  { color: #556677; font-size: 0.75rem; margin-top: 2px; }
    .about-xai-body  { color: #aabbcc; font-size: 0.84rem; line-height: 1.6; margin-bottom: 10px; }
    .about-bar-row { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
    .about-bar-label { width: 88px; text-align: right; color: #8899aa; font-size: 0.78rem;
                       flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .about-bar-track {
        flex: 1; height: 8px; background: #0f1a2e; border-radius: 99px; overflow: hidden;
    }
    .about-bar-fill { height: 100%; border-radius: 99px; }
    .about-bar-val { font-size: 0.78rem; width: 38px; flex-shrink: 0; }
    .about-token-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .about-token {
        font-size: 0.78rem; padding: 3px 10px; border-radius: 99px;
        font-family: monospace; font-weight: 600;
    }
    .tok-r { background: #3b0f0f; color: #f87171; }
    .tok-g { background: #0f2e12; color: #4ade80; }
    .tok-n { background: #1e2a3a; color: #8899aa; }
    .about-compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 14px 0 2.5rem; }
    .about-compare-card { border-radius: 12px; padding: 1rem 1.1rem; }
    .about-compare-title { font-size: 0.9rem; font-weight: 800; margin-bottom: 8px; }
    .about-compare-item { font-size: 0.82rem; color: #8899aa; margin: 5px 0;
                          display: flex; align-items: flex-start; gap: 8px; }
    .about-compare-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
    .about-risk-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 1rem 0 2.5rem; }
    .about-risk-card { border-radius: 12px; padding: 0.9rem 0.8rem; text-align: center; }
    .about-risk-label { font-size: 0.82rem; font-weight: 800; }
    .about-risk-range { font-size: 0.78rem; margin: 3px 0 8px; }
    .about-risk-desc  { font-size: 0.78rem; color: #8899aa; line-height: 1.5; }
    .about-url-grid {
        display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Section 1: Pipeline ──────────────────────────────
    st.markdown('<hr class="about-divider">', unsafe_allow_html=True)
    st.markdown('<span class="about-badge" style="background:#0f2a3a;color:#00d4ff;">🔍 How it works</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="about-h2">From text to verdict — the full pipeline</h2>', unsafe_allow_html=True)
    st.markdown('<p class="about-sub">Every claim you submit passes through four stages before a final decision is made.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-step-grid">
        <div class="about-step-card">
            <div class="about-step-num">01</div>
            <div class="about-step-title">📥 Input</div>
            <div class="about-step-desc">Paste a headline, article text, or a URL. The system extracts clean readable text for analysis.</div>
        </div>
        <div class="about-step-card">
            <div class="about-step-num">02</div>
            <div class="about-step-title">🤖 RoBERTa model</div>
            <div class="about-step-desc">A fine-tuned transformer reads the text and outputs a fake vs. real probability score from 0 to 1.</div>
        </div>
        <div class="about-step-card">
            <div class="about-step-num">03</div>
            <div class="about-step-title">🔬 XAI breakdown</div>
            <div class="about-step-desc">SHAP and LIME explain which exact words or tokens drove the model's decision — no black box.</div>
        </div>
        <div class="about-step-card">
            <div class="about-step-num">04</div>
            <div class="about-step-title">⚖️ Risk verdict</div>
            <div class="about-step-desc">Keyword signals and heuristics blend with the model score into a final LOW / MEDIUM / HIGH / CRITICAL risk level.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 2: SHAP & LIME ───────────────────────────
    st.markdown('<hr class="about-divider">', unsafe_allow_html=True)
    st.markdown('<span class="about-badge" style="background:#1a1040;color:#a78bfa;">📊 Explainability</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="about-h2">SHAP &amp; LIME — why the model decided what it did</h2>', unsafe_allow_html=True)
    st.markdown('<p class="about-sub">Both methods answer the same question — "which words mattered?" — but from different angles. Together they give you a complete picture.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-xai-grid">
        <div class="about-xai-card">
            <div class="about-xai-head">
                <div class="about-xai-icon" style="background:#1a1040;">🔢</div>
                <div>
                    <div class="about-xai-title">SHAP</div>
                    <div class="about-xai-full">SHapley Additive exPlanations</div>
                </div>
            </div>
            <div class="about-xai-body">
                Removes each token one at a time and measures how much the fake probability changes.
                A token that raises the score → fake signal. One that lowers it → real signal.
                The bar chart shows exactly how much each token contributed.
            </div>
            <div style="font-size:0.78rem;color:#556677;margin-bottom:8px;">Example — token contributions to fake probability</div>
            <div class="about-bar-row">
                <div class="about-bar-label">secret</div>
                <div class="about-bar-track"><div class="about-bar-fill" style="width:82%;background:#ef4444;"></div></div>
                <span class="about-bar-val" style="color:#f87171;">+0.42</span>
            </div>
            <div class="about-bar-row">
                <div class="about-bar-label">coverup</div>
                <div class="about-bar-track"><div class="about-bar-fill" style="width:62%;background:#ef4444;"></div></div>
                <span class="about-bar-val" style="color:#f87171;">+0.31</span>
            </div>
            <div class="about-bar-row">
                <div class="about-bar-label">according to</div>
                <div class="about-bar-track"><div class="about-bar-fill" style="width:52%;background:#22c55e;"></div></div>
                <span class="about-bar-val" style="color:#4ade80;">−0.28</span>
            </div>
            <div class="about-bar-row">
                <div class="about-bar-label">researchers</div>
                <div class="about-bar-track"><div class="about-bar-fill" style="width:38%;background:#22c55e;"></div></div>
                <span class="about-bar-val" style="color:#4ade80;">−0.19</span>
            </div>
        </div>
        <div class="about-xai-card">
            <div class="about-xai-head">
                <div class="about-xai-icon" style="background:#0a2818;">🧩</div>
                <div>
                    <div class="about-xai-title">LIME</div>
                    <div class="about-xai-full">Local Interpretable Model-agnostic Explanations</div>
                </div>
            </div>
            <div class="about-xai-body">
                Creates hundreds of slightly altered versions of your text (hiding different words), runs each through the model, and learns which words — if removed — flip the verdict from fake to real or vice versa.
            </div>
            <div style="font-size:0.78rem;color:#556677;margin-bottom:10px;">Example — word-level tags (red = toward fake, green = toward real)</div>
            <div class="about-token-row">
                <span class="about-token tok-r">5G</span>
                <span class="about-token tok-r">secretly</span>
                <span class="about-token tok-r">spreading</span>
                <span class="about-token tok-n">towers</span>
                <span class="about-token tok-n">the</span>
                <span class="about-token tok-g">study</span>
                <span class="about-token tok-g">confirmed</span>
                <span class="about-token tok-g">data</span>
                <span class="about-token tok-n">shows</span>
                <span class="about-token tok-r">control</span>
            </div>
            <div style="margin-top:12px;font-size:0.8rem;color:#556677;">Gray tokens had negligible influence on this prediction.</div>
        </div>
    </div>
    <div class="about-compare-grid">
        <div class="about-compare-card" style="background:#1a1040;border:1px solid #2d1b69;">
            <div class="about-compare-title" style="color:#a78bfa;">SHAP is better for…</div>
            <div class="about-compare-item"><div class="about-compare-dot" style="background:#7b2fff;"></div><span>Precise per-token numerical attribution</span></div>
            <div class="about-compare-item"><div class="about-compare-dot" style="background:#7b2fff;"></div><span>Understanding subword tokenizer output</span></div>
            <div class="about-compare-item"><div class="about-compare-dot" style="background:#7b2fff;"></div><span>Consistent additive scores across runs</span></div>
        </div>
        <div class="about-compare-card" style="background:#0a2818;border:1px solid #0f6e56;">
            <div class="about-compare-title" style="color:#4ade80;">LIME is better for…</div>
            <div class="about-compare-item"><div class="about-compare-dot" style="background:#22c55e;"></div><span>Human-readable whole-word explanations</span></div>
            <div class="about-compare-item"><div class="about-compare-dot" style="background:#22c55e;"></div><span>Quick "which word matters most" overview</span></div>
            <div class="about-compare-item"><div class="about-compare-dot" style="background:#22c55e;"></div><span>Model-agnostic — works on any classifier</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 3: Risk Levels ───────────────────────────
    st.markdown('<hr class="about-divider">', unsafe_allow_html=True)
    st.markdown('<span class="about-badge" style="background:#2a0a0a;color:#f87171;">⚠️ Risk levels</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="about-h2">What does the risk score mean?</h2>', unsafe_allow_html=True)
    st.markdown('<p class="about-sub">The model\'s fake probability is mapped to a four-tier risk level so you always know what action to take.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-risk-grid">
        <div class="about-risk-card" style="background:#0a2010;border:1px solid #1a5c2a;">
            <div class="about-risk-label" style="color:#4ade80;">🟢 LOW</div>
            <div class="about-risk-range" style="color:#22c55e;">below 50%</div>
            <div class="about-risk-desc">Content appears credible. Standard fact-checking before sharing is sufficient.</div>
        </div>
        <div class="about-risk-card" style="background:#2a1a00;border:1px solid #7a4500;">
            <div class="about-risk-label" style="color:#fbbf24;">🟡 MEDIUM</div>
            <div class="about-risk-range" style="color:#f59e0b;">50 – 65%</div>
            <div class="about-risk-desc">Exercise caution. Cross-check with trusted news sources before sharing.</div>
        </div>
        <div class="about-risk-card" style="background:#2a1000;border:1px solid #8b3e00;">
            <div class="about-risk-label" style="color:#fb923c;">🔴 HIGH</div>
            <div class="about-risk-range" style="color:#f97316;">65 – 80%</div>
            <div class="about-risk-desc">Do NOT share without verification from multiple credible outlets.</div>
        </div>
        <div class="about-risk-card" style="background:#2a0808;border:1px solid #8b1a1a;">
            <div class="about-risk-label" style="color:#f87171;">⛔ CRITICAL</div>
            <div class="about-risk-range" style="color:#ef4444;">above 80%</div>
            <div class="about-risk-desc">Highly likely misinformation. Flag this content immediately. Do not share.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 4: URL signals ───────────────────────────
    st.markdown('<hr class="about-divider">', unsafe_allow_html=True)
    st.markdown('<span class="about-badge" style="background:#0a1e2a;color:#38bdf8;">🔗 URL analysis</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="about-h2">URL mode — what signals are checked?</h2>', unsafe_allow_html=True)
    st.markdown('<p class="about-sub">When you paste a URL, domain reputation is checked before even reading the article. These four signals are evaluated instantly.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-url-grid">
        <div class="about-step-card">
            <div class="about-step-title">🎭 Typosquatting</div>
            <div class="about-step-desc">Detects lookalike domains (e.g. g00gle.com) using edit-distance matching against 50+ known brand names.</div>
        </div>
        <div class="about-step-card">
            <div class="about-step-title">🚩 Suspicious TLDs</div>
            <div class="about-step-desc">Flags high-risk extensions like .xyz .tk .click .buzz that are commonly used in spam and phishing campaigns.</div>
        </div>
        <div class="about-step-card">
            <div class="about-step-title">✅ Trusted TLDs</div>
            <div class="about-step-desc">.gov .edu .ac.in and institutional domains are auto-trusted and skip the model scoring entirely.</div>
        </div>
        <div class="about-step-card">
            <div class="about-step-title">📢 Clickbait path</div>
            <div class="about-step-desc">URL path words like "shocking", "leaked", "banned", "exposed" are detected as clickbait red flags.</div>
        </div>
    </div>
    <br><br>
    """, unsafe_allow_html=True)


if st.session_state.view == "home":
    render_home_page()
    st.stop()

# ── URL Analysis page ─────────────────────────────────────
def render_url_page():
    _hide_sidebar_for_landing()

    # Extra CSS for URL detector look
    st.markdown("""
    <style>
    .verdict-real-url{background:linear-gradient(135deg,#22c55e,#16a34a);border-radius:16px;padding:2rem;text-align:center;margin:1rem 0 1rem;}
    .verdict-fake-url{background:linear-gradient(135deg,#ef4444,#b91c1c);border-radius:16px;padding:2rem;text-align:center;margin:1rem 0 1rem;}
    .verdict-title-url{font-size:2rem;font-weight:700;color:white;margin:0.4rem 0;}
    .verdict-meta-url{color:rgba(255,255,255,0.85);font-size:0.95rem;}
    .stat-grid-url{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;}
    .stat-card-url{background:#1c2333;border:1px solid #30363d;border-radius:14px;padding:1.2rem 1rem;text-align:center;}
    .stat-value-url{font-size:1.6rem;font-weight:700;color:#00d4ff;}
    .stat-label-url{font-size:0.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.06em;margin-top:0.25rem;}
    .article-preview-url{background:#1c2333;border:1px solid #30363d;border-left:3px solid #00d4ff;border-radius:12px;padding:1.2rem;margin-bottom:1rem;}
    .article-preview-url h4{color:#e6edf3;margin:0 0 0.4rem;font-size:1rem;}
    .article-preview-url p{color:#8b949e;font-size:0.88rem;margin:0;line-height:1.55;}
    .section-title-url{font-size:1.1rem;font-weight:600;color:#e6edf3;margin:1.5rem 0 0.8rem;display:flex;align-items:center;gap:0.5rem;}
    .section-title-url::after{content:'';flex:1;height:1px;background:#30363d;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h1 class="welcome-title">🔗 URL Misinformation Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-sub">Full XAI analysis — phishing detection, RoBERTa, SHAP & LIME</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    _, col_main, _ = st.columns([0.3, 3, 0.3])
    with col_main:
        url_input = st.text_input(
            "URL", placeholder="https://www.example.com/news/article...",
            label_visibility="collapsed", key="url_detector_input"
        )
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            analyze_btn = st.button("🔍  Analyze URL Now", use_container_width=True, type="primary")
        with c2:
            clear_btn = st.button("🗑️  Clear", use_container_width=True)
        with c3:
            if st.button("🏠  Home", use_container_width=True):
                st.session_state.pop("url_last_result", None)
                st.session_state.view = "home"
                st.rerun()

        if clear_btn:
            st.session_state.pop("url_last_result", None)
            st.rerun()

        if analyze_btn and url_input.strip():
            raw_url = url_input.strip()
            is_valid, url_or_error = validate_url(raw_url)
            if not is_valid:
                st.error(f"⛔ Invalid URL: {url_or_error}")
                st.stop()

            url = url_or_error

            with st.spinner("🌐 Fetching article…"):
                article  = _scrape_article(url)
                features = _extract_url_features(url)
                url_type = _classify_url_type(features)

            scrape_failed = bool(article["error"]) or not article["full_text"]
            if not scrape_failed and _is_bot_page(article.get("full_text", "")):
                scrape_failed = True
                article["error"] = "Bot-protection page — could not extract article content"

            url_score = _score_url_only(features, url)
            lime_result = {"real":[],"fake":[],"error":"Skipped"}
            shap_result = {"tokens":[],"fake_shap":[],"real_shap":[],"error":"Skipped"}

            with st.spinner("🤖 Loading URL analysis model…"):
                tok_url, mdl_url = _load_url_model()

            if url_type == "institutional":
                result = {"label":"REAL","fake_prob":0.05,"real_prob":0.95,"confidence":0.95,
                          "note":"✅ Institutional / Educational / Govt domain — trusted by TLD or name.",
                          "url_type":url_type,"url_score":url_score}
            elif url_type == "credible" and not features.get("is_typosquat", False):
                result = {"label":"REAL","fake_prob":0.08,"real_prob":0.92,"confidence":0.92,
                          "note":"✅ Verified credible domain.",
                          "url_type":url_type,"url_score":url_score,
                          "reasons":["✅ Domain is on the verified credible sources list",
                                     "✅ No suspicious URL signals detected"]}
            elif features.get("is_typosquat", False):
                result = {"label":url_score["label"],"fake_prob":url_score["fake_score"],
                          "real_prob":url_score["real_score"],"confidence":url_score["confidence"],
                          "note":"⚠️ Typosquat/phishing domain detected.",
                          "url_type":url_type,"url_score":url_score,"reasons":url_score["reasons"]}
            elif scrape_failed:
                result = {"label":url_score["label"],"fake_prob":url_score["fake_score"],
                          "real_prob":url_score["real_score"],"confidence":url_score["confidence"],
                          "note":f"📡 URL-only analysis (page not loaded).",
                          "url_type":url_type,"url_score":url_score,"reasons":url_score["reasons"]}
            else:
                text = article["full_text"]
                with st.spinner("🤖 Running RoBERTa on article text…"):
                    ai_result = _predict_text_url(text, tok_url, mdl_url)
                blended_fake = 0.70 * ai_result["fake_prob"] + 0.30 * url_score["fake_score"]
                blended_real = 1.0 - blended_fake
                if features.get("is_typosquat"):
                    blended_fake = max(blended_fake, 0.75)
                    blended_real = 1.0 - blended_fake
                if features["is_known_credible"] and not features.get("is_typosquat"):
                    bump = 0.20 if ai_result["confidence"] < 0.70 else 0.10
                    blended_fake = max(0.05, blended_fake - bump)
                    blended_real = min(0.95, blended_real + bump)
                result = {"label":"FAKE" if blended_fake > blended_real else "REAL",
                          "fake_prob":round(blended_fake,4),"real_prob":round(blended_real,4),
                          "confidence":round(max(blended_fake, blended_real),4),
                          "note":"","url_type":url_type,"url_score":url_score,
                          "reasons":url_score["reasons"],"ai_raw":ai_result}
                with st.spinner("🔬 Running LIME (~30 sec)…"):
                    lime_result = _run_lime_url(text, tok_url, mdl_url)
                with st.spinner("🧬 Computing SHAP values…"):
                    shap_result = _run_shap_url(text, tok_url, mdl_url)

            st.session_state["url_last_result"] = {
                "url":url,"article":article,"result":result,
                "features":features,"lime":lime_result,"shap":shap_result,
                "scrape_failed":scrape_failed,
            }

        # ── Show results ──────────────────────────────────────
        if "url_last_result" in st.session_state:
            d        = st.session_state["url_last_result"]
            result   = d["result"]
            features = d["features"]
            article  = d["article"]
            url      = d["url"]

            risk_color_map = {"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#ef4444"}
            fp = result["fake_prob"]
            risk_lbl = "LOW" if fp < 0.35 else ("MEDIUM" if fp < 0.65 else "HIGH")
            risk_color = risk_color_map[risk_lbl]

            if result["label"] == "REAL":
                st.markdown(f"""<div class="verdict-real-url">
                    <div style="font-size:2.5rem">✅</div>
                    <div class="verdict-title-url">REAL NEWS</div>
                    <div class="verdict-meta-url">Confidence: <b>{result['confidence']*100:.1f}%</b>
                    &nbsp;|&nbsp; Risk: <span style="color:{risk_color};font-weight:700">● {risk_lbl}</span></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="verdict-fake-url">
                    <div style="font-size:2.5rem">❌</div>
                    <div class="verdict-title-url">FAKE / SUSPICIOUS</div>
                    <div class="verdict-meta-url">Confidence: <b>{result['confidence']*100:.1f}%</b>
                    &nbsp;|&nbsp; Risk: <span style="color:{risk_color};font-weight:700">● {risk_lbl}</span></div>
                </div>""", unsafe_allow_html=True)

            # Stat grid
            st.markdown(f"""<div class="stat-grid-url">
                <div class="stat-card-url"><div class="stat-value-url">{result["label"]}</div><div class="stat-label-url">Verdict</div></div>
                <div class="stat-card-url"><div class="stat-value-url" style="color:{risk_color}">{risk_lbl}</div><div class="stat-label-url">Risk Level</div></div>
                <div class="stat-card-url"><div class="stat-value-url">{result['fake_prob']*100:.1f}%</div><div class="stat-label-url">Fake Probability</div></div>
                <div class="stat-card-url"><div class="stat-value-url">{result['confidence']*100:.1f}%</div><div class="stat-label-url">Confidence</div></div>
            </div>""", unsafe_allow_html=True)

            # Signals box
            reasons = result.get("reasons", []) or []
            if reasons:
                box_color = "#ef4444" if result["label"] == "FAKE" else "#22c55e"
                box_label = "🚨 Suspicious Signals" if result["label"] == "FAKE" else "✅ Trust Signals"
                items_html = "".join(
                    f'<div style="margin:0.35rem 0;font-size:0.88rem;color:#e6edf3;">{r}</div>'
                    for r in reasons
                )
                st.markdown(f"""<div style="background:#1c2333;border:1px solid #30363d;border-left:3px solid {box_color};
                            border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
                    <div style="font-weight:700;color:{box_color};margin-bottom:0.6rem;">{box_label}</div>
                    {items_html}
                </div>""", unsafe_allow_html=True)

            if result.get("note"):
                st.info(result["note"])

            if result.get("ai_raw"):
                ai = result["ai_raw"]
                st.caption(f"🤖 RoBERTa raw: FAKE {ai['fake_prob']*100:.1f}% · REAL {ai['real_prob']*100:.1f}% → blended with URL signals → Final FAKE {result['fake_prob']*100:.1f}%")

            # Article preview
            if article.get("title") and not _is_bot_page(article.get("title", "")):
                snippet = (article.get("description") or article.get("body",""))[:280]
                st.markdown(f"""<div class="article-preview-url">
                    <h4>📰 {article['title']}</h4>
                    <p>{snippet}…</p>
                </div>""", unsafe_allow_html=True)

            # Gauges
            st.markdown('<div class="section-title-url">📊 Probability Gauges</div>', unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1: st.plotly_chart(_url_gauge_chart(result["fake_prob"],"FAKE Probability","#ef4444"), use_container_width=True)
            with g2: st.plotly_chart(_url_gauge_chart(result["real_prob"],"REAL Probability","#22c55e"), use_container_width=True)

            # URL structure chart
            st.markdown('<div class="section-title-url">🔗 URL Structure Analysis</div>', unsafe_allow_html=True)
            st.plotly_chart(_url_structure_chart(features), use_container_width=True)
            with st.expander("📋 Raw URL Feature Details"):
                st.dataframe(
                    pd.DataFrame([{"Feature":k.replace("_"," ").title(),"Value":str(v)} for k,v in features.items()]),
                    hide_index=True, use_container_width=True
                )

            # LIME
            lime_data = d["lime"]
            if not lime_data.get("error") and (lime_data.get("real") or lime_data.get("fake")):
                st.markdown('<div class="section-title-url">🔬 LIME Explainability</div>', unsafe_allow_html=True)
                l1, l2 = st.columns(2)
                with l1:
                    fig = _lime_bar_url(lime_data, "REAL")
                    if fig: st.plotly_chart(fig, use_container_width=True)
                with l2:
                    fig = _lime_bar_url(lime_data, "FAKE")
                    if fig: st.plotly_chart(fig, use_container_width=True)

            # SHAP
            shap_data = d["shap"]
            if not shap_data.get("error") and shap_data.get("tokens"):
                st.markdown('<div class="section-title-url">🧬 SHAP Token Attribution</div>', unsafe_allow_html=True)
                fig = _shap_bar_url(shap_data)
                if fig: st.plotly_chart(fig, use_container_width=True)

            # PDF download
            st.markdown('<div class="section-title-url">📄 Report</div>', unsafe_allow_html=True)
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    path = _generate_url_pdf(url, result, features, article)
                with open(path,"rb") as f:
                    st.download_button("⬇️  Download PDF Report", f,
                        file_name="url_analysis_report.pdf", mime="application/pdf", use_container_width=True)
                os.unlink(path)

        elif not url_input.strip() and "url_last_result" not in st.session_state:
            st.markdown("""
            <div style="text-align:center;color:#8b949e;padding:3rem 0;">
                <div style="font-size:3.5rem">🔗</div>
                <div style="font-size:1.1rem;margin-top:1rem;">
                    Paste a news URL above and click <b style="color:#00d4ff">Analyze URL Now</b>
                </div>
                <div style="font-size:0.88rem;margin-top:0.5rem;">
                    Detects phishing • Runs RoBERTa on article text • SHAP & LIME explainability
                </div>
            </div>""", unsafe_allow_html=True)


if st.session_state.view == "url":
    render_url_page()
    st.stop()

# ── Keyword lists ────────────────────────────────────────
FAKE_KEYWORDS = [
    "secret", "secretly", "hoax", "conspiracy", "hidden", "suppressed",
    "cover up", "coverup", "deep state", "illuminati", "reptilian",
    "microchip", "mind control", "chemtrail", "flat earth", "crisis actor",
    "false flag", "staged", "aliens built", "miracle cure", "big pharma",
    "cures cancer", "guaranteed", "overnight billionaire", "bleach cure",
    "aliens living", "moon hollow", "never existed", "fabricated",
    "lizard people", "new world order", "depopulation", "poison water",
    "fluoride mind", "satellites read", "weather control", "earthquake machine",
    "pre-determined", "fossils are fake", "dinosaurs never", "evolution hoax",
    "government hiding", "wake up", "sheeple", "100% success",
    "100% guaranteed", "doctors dont want", "banned by", "censored",
    "whistleblower reveals", "mainstream media lies", "plandemic",
    "fake virus", "invented covid", "5g spread", "towers spread",
    "secretly spreading", "control population", "mind reading",
    "thought control", "secret society", "they dont want you to know",
    "doctors warn", "breaking news", "health risks", "serious health risks",
    "water intoxication", "claims dozens", "clickbait", "shocking",
    "misleading", "life-threatening", "deadly",
]

REAL_KEYWORDS = [
    "according to", "study shows", "researchers found", "data shows",
    "percent", "federal reserve", "nasa launched", "scientists confirmed",
    "government approved", "reported", "announced", "published",
    "clinical trial", "fda approved", "statistics", "survey results",
    "economists predict", "labor department", "health department",
    "university researchers", "new research", "findings show",
    "experts say", "officials said", "department of", "bureau of",
    "increased by", "decreased by", "rose by", "fell by",
]

def get_keywords(text):
    text_lower = text.lower()
    fake_hits = [kw for kw in FAKE_KEYWORDS if kw in text_lower]
    real_hits = [kw for kw in REAL_KEYWORDS if kw in text_lower]
    return fake_hits, real_hits

# ── Batch file helpers ────────────────────────────────────
BATCH_FILE_TYPES = ["csv", "tsv", "txt", "text", "tab", "json", "jsonl", "ndjson", "md", "log"]

TEXT_COLUMN_HINTS = [
    "statement", "text", "claim", "content", "headline", "title", "body",
    "tweet", "news", "article", "review", "message", "description", "sentence",
    "post", "comment", "summary", "news_text", "news_body", "article_body",
    "news_title", "full_text", "document", "input", "data",
]


def _normalize_df_columns(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def detect_text_columns(df):
    df = _normalize_df_columns(df)
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    detected = []
    for hint in TEXT_COLUMN_HINTS:
        if hint in lower_map and lower_map[hint] not in detected:
            detected.append(lower_map[hint])
    scored = []
    for c in cols:
        series = df[c].dropna().astype(str).str.strip()
        series = series[series != ""]
        if len(series) == 0:
            continue
        avg_len = series.str.len().mean()
        numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
        if numeric_ratio > 0.85 and avg_len < 25:
            continue
        scored.append((c, avg_len))
    scored.sort(key=lambda x: -x[1])
    for c, avg_len in scored:
        if c not in detected and avg_len >= 10:
            detected.append(c)
    if not detected:
        for c in cols:
            if df[c].dtype == object or str(df[c].dtype) == "string":
                detected.append(c)
    if not detected and cols:
        detected = [cols[0]]
    return detected


def _read_file_bytes(uploaded_file):
    uploaded_file.seek(0)
    return uploaded_file.read()


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_uploaded_csv(uploaded_file, sep=","):
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            uploaded_file.seek(0)
            return _normalize_df_columns(pd.read_csv(uploaded_file, sep=sep, encoding=encoding))
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    uploaded_file.seek(0)
    return _normalize_df_columns(
        pd.read_csv(uploaded_file, sep=sep, encoding="utf-8", on_bad_lines="skip")
    )


def read_uploaded_text_lines(uploaded_file):
    content = _decode_bytes(_read_file_bytes(uploaded_file))
    lines = []
    for ln in content.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            lines.append(ln)
    if not lines:
        raise ValueError("Text file has no non-empty lines.")
    return pd.DataFrame({"line": range(1, len(lines) + 1), "text": lines})


def read_uploaded_jsonl(uploaded_file):
    content = _decode_bytes(_read_file_bytes(uploaded_file))
    records = []
    for ln in content.splitlines():
        ln = ln.strip()
        if ln:
            records.append(json.loads(ln))
    if not records:
        raise ValueError("JSONL file is empty.")
    return _normalize_df_columns(pd.DataFrame(records))


def read_uploaded_json(uploaded_file):
    data = json.loads(_decode_bytes(_read_file_bytes(uploaded_file)))
    if isinstance(data, list):
        return _normalize_df_columns(pd.DataFrame(data))
    if isinstance(data, dict):
        for key in ("data", "records", "items", "results", "articles", "posts", "rows"):
            if key in data and isinstance(data[key], list):
                return _normalize_df_columns(pd.DataFrame(data[key]))
        return _normalize_df_columns(pd.DataFrame([data]))
    raise ValueError("JSON must be a list or an object containing a list field.")


def read_uploaded_batch_file(uploaded_file):
    name = uploaded_file.name.lower()
    ext = os.path.splitext(name)[1]
    if ext == ".csv":
        return read_uploaded_csv(uploaded_file), "tabular"
    if ext in (".tsv", ".tab"):
        return read_uploaded_csv(uploaded_file, sep="\t"), "tabular"
    if ext in (".txt", ".text", ".md", ".log"):
        return read_uploaded_text_lines(uploaded_file), "lines"
    if ext in (".jsonl", ".ndjson"):
        return read_uploaded_jsonl(uploaded_file), "tabular"
    if ext == ".json":
        return read_uploaded_json(uploaded_file), "tabular"
    try:
        uploaded_file.seek(0)
        df = read_uploaded_csv(uploaded_file)
        if len(df.columns) >= 1 and len(df) > 0:
            return df, "tabular"
    except Exception:
        pass
    uploaded_file.seek(0)
    try:
        df = read_uploaded_csv(uploaded_file, sep="\t")
        if len(df.columns) > 1 and len(df) > 0:
            return df, "tabular"
    except Exception:
        pass
    uploaded_file.seek(0)
    return read_uploaded_text_lines(uploaded_file), "lines"


# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(SAVED_PATH + "\\config.json"):
        tokenizer = RobertaTokenizer.from_pretrained(SAVED_PATH)
        model     = RobertaForSequenceClassification.from_pretrained(SAVED_PATH)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        model     = RobertaForSequenceClassification.from_pretrained(
                        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    model.eval()
    return model, tokenizer


# ── Raw model predict ─────────────────────────────────────
def raw_predict_fake(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=MAX_LEN)
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).numpy()[0]
    gc.collect()
    return float(probs[0]), float(probs[1])

def raw_predict_proba(texts, model, tokenizer):
    results = []
    for t in texts:
        pr, pf = raw_predict_fake(t, model, tokenizer)
        results.append([pr, pf])
        gc.collect()
    return np.array(results)


# ── Hybrid predict ────────────────────────────────────────
def predict(text, model, tokenizer):
    from claim_heuristics import check_misleading_claim

    prob_real, prob_fake = raw_predict_fake(text, model, tokenizer)
    fake_hits, real_hits = get_keywords(text)
    fake_boost = min(len(fake_hits) * 0.18, 0.55)
    real_boost = min(len(real_hits) * 0.10, 0.35)

    heuristic = check_misleading_claim(text)
    if heuristic and heuristic["verdict"] == "FAKE":
        fake_boost = max(fake_boost, float(heuristic["confidence"]) - prob_fake)

    adj_fake = prob_fake + fake_boost - real_boost
    adj_real = prob_real + real_boost - fake_boost
    total    = abs(adj_fake) + abs(adj_real)
    adj_fake = max(0.01, min(0.99, adj_fake / total))
    adj_real = 1 - adj_fake
    pred = {
        "label":               "Fake" if adj_fake > 0.5 else "Real",
        "prob_real":           round(adj_real, 4),
        "prob_fake":           round(adj_fake, 4),
        "fake_keywords_found": fake_hits,
        "real_keywords_found": real_hits,
    }
    if heuristic:
        pred["heuristic_verdict"]     = heuristic["verdict"]
        pred["heuristic_explanation"] = heuristic["explanation"]
    return pred


# ── SHAP ──────────────────────────────────────────────────
def clean_token(t):
    return t.replace("\u0120", "").replace("\u010a", "").replace("Ġ", "").replace("Ċ", "").strip()

def compute_shap(text, model, tokenizer):
    tokens = tokenizer.tokenize(text)
    _, base_prob = raw_predict_fake(text, model, tokenizer)
    shap_vals = []
    for i in range(len(tokens)):
        masked = tokenizer.convert_tokens_to_string(tokens[:i] + tokens[i+1:])
        if not masked.strip():
            shap_vals.append(0.0)
            continue
        _, mp = raw_predict_fake(masked, model, tokenizer)
        shap_vals.append(round(base_prob - mp, 4))
        gc.collect()
    clean_tokens = [clean_token(t) for t in tokens]
    return list(zip(clean_tokens, shap_vals))


# ── LIME ──────────────────────────────────────────────────
def compute_lime(text, model, tokenizer):
    explainer = LimeTextExplainer(class_names=["Real", "Fake"], random_state=42)
    def predictor(texts):
        return raw_predict_proba(texts, model, tokenizer)
    exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=50)
    return exp.as_list()


# ── Explanation generator ─────────────────────────────────
def generate_explanation(text, pred, risk, shap_data, lime_data, fake_kws, real_kws):
    label     = pred["label"]
    prob_fake = pred["prob_fake"]
    risk_lvl  = risk["risk_level"]
    top_shap_fake = [(t, v) for t, v in sorted(shap_data, key=lambda x: x[1], reverse=True) if v > 0][:3]
    top_shap_real = [(t, v) for t, v in sorted(shap_data, key=lambda x: x[1]) if v < 0][:3]
    top_lime_fake = [(w, wt) for w, wt in sorted(lime_data, key=lambda x: x[1], reverse=True) if wt > 0][:3]
    top_lime_real = [(w, wt) for w, wt in sorted(lime_data, key=lambda x: x[1]) if wt < 0][:3]
    lines = []
    if label == "Fake":
        lines.append(f"This content has been classified as FAKE NEWS with {prob_fake:.1%} fake probability and {risk_lvl} risk level.")
    else:
        lines.append(f"This content has been classified as REAL NEWS with {1-prob_fake:.1%} real probability and {risk_lvl} risk level.")
    if pred.get("realtime_verdict") and pred["realtime_verdict"] != "UNVERIFIABLE":
        lines.append(
            f"Verification: {pred['realtime_verdict']} "
            f"({pred['realtime_confidence']:.0%} confidence). "
            f"{pred.get('realtime_explanation', '')}"
        )
        for link in pred.get("proof_links", [])[:1]:
            url = link.get("url", "")
            if url and "wikipedia.org" in url.lower():
                lines.append(f"Wikipedia: {link.get('title', 'Reference')} — {url}")
    lines.append("")
    if fake_kws:
        lines.append(f"Suspicious keywords detected: {', '.join(fake_kws[:5])}.")
        lines.append("These words are commonly associated with misinformation, conspiracy theories, or unverified claims.")
    if real_kws:
        lines.append(f"Credibility indicators found: {', '.join(real_kws[:5])}.")
        lines.append("These phrases suggest factual reporting based on data or official sources.")
    lines.append("")
    if top_shap_fake:
        lines.append(f"SHAP Analysis: The tokens {', '.join([f'{t}' for t, _ in top_shap_fake])} contributed most strongly toward the FAKE classification.")
    if top_shap_real:
        lines.append(f"The tokens {', '.join([f'{t}' for t, _ in top_shap_real])} pushed the model toward REAL classification.")
    lines.append("")
    if top_lime_fake:
        lines.append(f"LIME Analysis: The words {', '.join([f'{w}' for w, _ in top_lime_fake])} were the strongest indicators of misinformation.")
    if top_lime_real:
        lines.append(f"The words {', '.join([f'{w}' for w, _ in top_lime_real])} supported the credibility of this content.")
    lines.append("")
    recommendations = {
        "LOW":      "Recommendation: Content appears credible. Standard fact-checking is sufficient before sharing.",
        "MEDIUM":   "Recommendation: Exercise caution. Cross-check with trusted news sources before sharing.",
        "HIGH":     "Recommendation: Do NOT share without thorough verification from multiple credible sources.",
        "CRITICAL": "Recommendation: This content is highly likely to be misinformation. Flag and do not share.",
    }
    lines.append(recommendations[risk_lvl])
    return "\n".join(lines)


# ── Charts ────────────────────────────────────────────────
def shap_chart(token_shap):
    top    = sorted(token_shap, key=lambda x: abs(x[1]), reverse=True)[:10]
    tokens = [t for t, _ in top]
    values = [x[1] for x in top]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    bars = ax.barh(tokens, values, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#556677", linewidth=1)
    ax.set_xlabel("SHAP Value  (red = toward Fake,  green = toward Real)", color="#8899aa", fontsize=9)
    ax.set_title("SHAP: Which tokens drove the prediction?", color="#00d4ff", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aabbcc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d4a6e")
    ax.spines["bottom"].set_color("#2d4a6e")
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(val + (0.0002 if val >= 0 else -0.0002),
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center",
                ha="left" if val >= 0 else "right",
                color="#aabbcc", fontsize=8)
    plt.tight_layout()
    return fig

def lime_chart(lime_list):
    words   = [x[0] for x in lime_list]
    weights = [x[1] for x in lime_list]
    colors  = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    ax.barh(words, weights, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#556677", linewidth=1)
    ax.set_xlabel("LIME Weight  (red = toward Fake,  green = toward Real)", color="#8899aa", fontsize=9)
    ax.set_title("LIME: Which words matter most?", color="#00d4ff", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aabbcc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d4a6e")
    ax.spines["bottom"].set_color("#2d4a6e")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def gauge_chart(prob_fake):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    colors = ["#27ae60", "#f1c40f", "#e67e22", "#c0392b"]
    bounds = [0, 0.50, 0.65, 0.80, 1.0]
    for i in range(4):
        start = np.pi * (1 - bounds[i+1])
        end   = np.pi * (1 - bounds[i])
        theta = np.linspace(start, end, 100)
        ax.fill_between(theta, 0.55, 1.0, color=colors[i], alpha=0.85)
    angle = np.pi * (1 - prob_fake)
    ax.annotate("", xy=(angle, 0.8), xytext=(np.pi / 2, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5))
    ax.set_ylim(0, 1)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis("off")
    ax.set_title(f"{prob_fake:.1%} Fake Probability", color="#00d4ff", pad=8, fontsize=11, fontweight="bold")
    for angle_d, label in [(180, "0%"), (135, "25%"), (90, "50%"), (45, "75%"), (0, "100%")]:
        angle_r = np.radians(angle_d)
        ax.text(angle_r, 1.15, label, ha="center", va="center", color="#8899aa", fontsize=7)
    plt.tight_layout()
    return fig

def probability_bar(prob_real, prob_fake):
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    ax.barh(0, prob_real, color="#27ae60", height=0.5)
    ax.barh(0, prob_fake, left=prob_real, color="#e74c3c", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(prob_real / 2, 0, f"Real\n{prob_real:.1%}",
            ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.text(prob_real + prob_fake / 2, 0, f"Fake\n{prob_fake:.1%}",
            ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    plt.tight_layout()
    return fig


def render_analysis_results(
    text, pred, risk, shap_data, lime_data, fake_kws, real_kws,
    run_shap, run_lime, run_semantic, run_explain, report_btn=False,
):
    st.markdown("---")
    verdict_class = "verdict-fake" if pred["label"] == "Fake" else "verdict-real"
    verdict_icon  = "🚨" if pred["label"] == "Fake" else "✅"
    st.markdown(f"""
    <div class="{verdict_class}">
        <h1>{verdict_icon} {pred['label'].upper()} NEWS</h1>
        <p>Confidence: {risk['confidence']} &nbsp;|&nbsp; Risk Level: {risk['icon']} {risk['risk_level']}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    if pred.get("heuristic_verdict") == "FAKE" and pred.get("heuristic_explanation"):
        st.markdown(f"""
        <div style='background:#c0392b22; border-left:4px solid #c0392b;
                    padding:12px 16px; border-radius:6px; margin-bottom:12px'>
            <span style='color:#ffffff;font-weight:700;font-size:1rem'>
                ⚠️ Context Check: MISLEADING / FAKE
            </span><br>
            <span style='color:#ddeeff;font-size:0.9rem'>{pred["heuristic_explanation"]}</span>
        </div>""", unsafe_allow_html=True)

    if pred.get("realtime_verdict") and pred["realtime_verdict"] != "UNVERIFIABLE":
        rv = pred["realtime_verdict"]
        rc = pred.get("realtime_confidence", 0)
        re = pred.get("realtime_explanation", "")
        rt_color = {"FAKE": "#c0392b", "REAL": "#27ae60"}.get(rv, "#7f8c8d")
        rt_icon  = {"FAKE": "🚨", "REAL": "✅"}.get(rv, "❓")
        why_label = "Why this is FAKE" if rv == "FAKE" else "Why this is REAL"
        st.markdown(f"""
        <div style='background:{rt_color}22; border-left:4px solid {rt_color};
                    padding:12px 16px; border-radius:6px; margin-bottom:12px'>
            <span style='color:#ffffff;font-weight:700;font-size:1rem'>
                {rt_icon} {why_label}
            </span>
            &nbsp;<span style='color:#aabbcc;font-size:0.85rem'>(verification confidence: {rc:.0%})</span><br>
            <span style='color:#ddeeff;font-size:0.9rem'>{re}</span>
        </div>""", unsafe_allow_html=True)

        proof_links = [
            p for p in pred.get("proof_links", [])
            if p.get("url") and "wikipedia.org" in str(p["url"]).lower()
        ]
        if proof_links:
            st.markdown('<div class="section-header">Wikipedia Proof</div>', unsafe_allow_html=True)
            item = proof_links[0]
            title = item.get("title", "Wikipedia article")
            url = item.get("url", "")
            st.markdown(
                f'<div style="margin:6px 0;padding:10px 14px;background:#1e2a3a;'
                f'border-radius:6px;border-left:3px solid #00d4ff">'
                f'<a href="{url}" target="_blank" style="color:#00d4ff;font-weight:600;font-size:1rem">'
                f'{title}</a></div>',
                unsafe_allow_html=True,
            )
            if pred.get("wikipedia_extract"):
                st.caption(pred["wikipedia_extract"])
        st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h2>{pred["label"]}</h2><p>Verdict</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h2>{risk["icon"]}</h2><p>{risk["risk_level"]} Risk</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h2>{pred["prob_fake"]:.1%}</h2><p>Fake Probability</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h2>{risk["confidence"]}</h2><p>Confidence</p></div>', unsafe_allow_html=True)
    st.markdown("")

    col_g, col_p = st.columns([1, 2])
    with col_g:
        st.markdown('<div class="section-header">Risk Gauge</div>', unsafe_allow_html=True)
        st.pyplot(gauge_chart(pred["prob_fake"]))
    with col_p:
        st.markdown('<div class="section-header">Probability Split</div>', unsafe_allow_html=True)
        st.pyplot(probability_bar(pred["prob_real"], pred["prob_fake"]))
        st.markdown("")
        color_map = {"LOW": "#27ae60", "MEDIUM": "#f1c40f", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"}
        descriptions = {
            "LOW":      "Low misinformation risk. Content appears credible.",
            "MEDIUM":   "Moderate risk detected. Manual review recommended.",
            "HIGH":     "High risk. Do not share without verification.",
            "CRITICAL": "Critical risk. Likely misinformation. Flag immediately.",
        }
        st.markdown(f"""
        <div style='background:{color_map[risk["risk_level"]]}22;
                    border-left:4px solid {color_map[risk["risk_level"]]};
                    padding:12px; border-radius:6px; color:#ddeeff;'>
            {risk["icon"]} <b style='color:#ffffff'>{risk["risk_level"]}</b> — {descriptions[risk["risk_level"]]}
        </div>""", unsafe_allow_html=True)

    if fake_kws or real_kws:
        st.markdown("---")
        st.markdown('<div class="section-header">Keywords Detected</div>', unsafe_allow_html=True)
        if fake_kws:
            st.markdown("<p style='color:#ffaaaa;font-weight:600'>Suspicious keywords:</p>", unsafe_allow_html=True)
            st.markdown("".join([f'<span class="keyword-fake">{k}</span>' for k in fake_kws]), unsafe_allow_html=True)
        if real_kws:
            st.markdown("<p style='color:#aaffaa;font-weight:600'>Credibility keywords:</p>", unsafe_allow_html=True)
            st.markdown("".join([f'<span class="keyword-real">{k}</span>' for k in real_kws]), unsafe_allow_html=True)

    st.markdown("---")
    if run_shap or run_lime:
        st.markdown('<div class="section-header">Explainability Analysis</div>', unsafe_allow_html=True)
        col_s, col_l = st.columns(2)
        if run_shap and shap_data:
            with col_s:
                st.markdown("<p style='color:#00d4ff;font-weight:700'>SHAP — Token Level</p>", unsafe_allow_html=True)
                st.pyplot(shap_chart(shap_data))
                with st.expander("Raw SHAP values"):
                    sdf = pd.DataFrame(shap_data, columns=["Token", "SHAP Value"])
                    sdf["Direction"] = sdf["SHAP Value"].apply(lambda v: ">> Fake" if v > 0 else ">> Real")
                    st.dataframe(sdf.sort_values("SHAP Value", key=abs, ascending=False), use_container_width=True)
        if run_lime and lime_data:
            with col_l:
                st.markdown("<p style='color:#00d4ff;font-weight:700'>LIME — Word Level</p>", unsafe_allow_html=True)
                st.pyplot(lime_chart(lime_data))
                with st.expander("Raw LIME values"):
                    ldf = pd.DataFrame(lime_data, columns=["Word", "Weight"])
                    ldf["Direction"] = ldf["Weight"].apply(lambda v: ">> Fake" if v > 0 else ">> Real")
                    st.dataframe(ldf.sort_values("Weight", key=abs, ascending=False), use_container_width=True)

    if run_explain and shap_data and lime_data:
        st.markdown("---")
        st.markdown('<div class="section-header">AI Explanation</div>', unsafe_allow_html=True)
        explanation = generate_explanation(text, pred, risk, shap_data, lime_data, fake_kws, real_kws)
        st.markdown(f'<div class="explain-box">{explanation.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    if run_semantic:
        st.markdown("---")
        st.markdown('<div class="section-header">Semantic Features</div>', unsafe_allow_html=True)
        _spacy_ok    = False
        _textblob_ok = False

        try:
            import spacy as _spacy
            _spacy_ok = True
        except ImportError:
            pass

        try:
            from textblob import TextBlob as _TextBlob
            _textblob_ok = True
        except ImportError:
            pass

        if not _spacy_ok or not _textblob_ok:
            missing = []
            if not _spacy_ok:    missing.append("`pip install spacy`  then  `python -m spacy download en_core_web_sm`")
            if not _textblob_ok: missing.append("`pip install textblob`  then  `python -m textblob.download_corpora`")
            st.warning(
                "Semantic analysis requires extra packages.\n\n"
                "Run these in your terminal and **restart the app**:\n\n"
                + "\n\n".join(missing)
            )
        else:
            try:
                _nlp = _spacy.load("en_core_web_sm")
            except OSError:
                st.warning(
                    "spaCy model **en_core_web_sm** not found.\n\n"
                    "Run: `python -m spacy download en_core_web_sm` then restart."
                )
                _nlp = None

            if '_nlp' in dir() and _nlp is not None:
                try:
                    doc  = _nlp(text)
                    blob = _TextBlob(text)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Sentiment",    f"{blob.sentiment.polarity:.3f}")
                    s2.metric("Subjectivity", f"{blob.sentiment.subjectivity:.3f}")
                    s3.metric("Word Count",   len([t for t in doc if t.is_alpha]))
                    s4.metric("Sentences",    len(list(doc.sents)))
                    entities = [(e.text, e.label_) for e in doc.ents]
                    if entities:
                        st.markdown(
                            "<p style='color:#ddeeff'><b>Named Entities:</b> "
                            + "  ".join([f"<code>{e[0]}</code> ({e[1]})" for e in entities])
                            + "</p>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("<p style='color:#8899aa'>Named Entities: None detected</p>", unsafe_allow_html=True)
                except Exception as _sem_err:
                    st.warning(f"Semantic analysis error: {_sem_err}")


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0'>
        <div style='font-size:3rem'>🛡️</div>
        <div style='color:#00d4ff; font-weight:700; font-size:1.1rem'>XAI Detector</div>
        <div style='color:#aabbcc; font-size:0.75rem'>Powered by RoBERTa</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.view = "home"
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    run_shap     = st.toggle("SHAP Analysis",     value=True)
    run_lime     = st.toggle("LIME Analysis",     value=True)
    run_semantic = st.toggle("Semantic Features", value=True)
    run_explain  = st.toggle("AI Explanation",    value=True)

    st.markdown("---")
    st.markdown("### 🎯 Risk Levels")
    st.markdown("""
    <div style='color:#ddeeff; font-size:0.9rem; line-height:2'>
        🟢 &nbsp;<b style='color:#27ae60'>LOW</b> &nbsp;— below 50%<br>
        🟡 &nbsp;<b style='color:#f1c40f'>MEDIUM</b> &nbsp;— 50–65%<br>
        🔴 &nbsp;<b style='color:#e67e22'>HIGH</b> &nbsp;— 65–80%<br>
        ⛔ &nbsp;<b style='color:#e74c3c'>CRITICAL</b> &nbsp;— above 80%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    total = len(st.session_state.history)
    fakes = sum(1 for h in st.session_state.history if h["verdict"] == "Fake")
    reals = total - fakes
    st.metric("Total Analyzed", total)
    c1, c2 = st.columns(2)
    c1.metric("Fake", fakes)
    c2.metric("Real", reals)
    if total > 0 and st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.batch_results = None
        st.rerun()


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
st.markdown('<h1 class="main-title">XAI Misinformation Detector</h1>', unsafe_allow_html=True)
st.markdown("")

model, tokenizer = load_model()

if st.session_state.view not in ("text", "home", "url"):
    st.session_state.view = "text"

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Text", "📂 Batch Analysis", "📜 History"])

# ════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Enter Text to Analyze</div>', unsafe_allow_html=True)

    examples = [
        "Select an example...",
        "5G towers are secretly spreading viruses to control the population.",
        "The Federal Reserve raised interest rates by 0.25 percent.",
        "Scientists confirmed water ice deposits near the lunar south pole.",
        "Drinking bleach can cure the common cold according to a secret study.",
        "NASA successfully launched its new Mars exploration rover yesterday.",
        "The moon landing was staged in a Hollywood studio by the CIA.",
        "Climate change is a hoax invented by foreign governments.",
        "The government approved a new infrastructure bill worth 1.2 trillion dollars.",
        "Aliens are secretly living inside the moon confirmed by NASA.",
        "A student became a billionaire overnight by liking social media posts.",
        "Eating chocolate every hour guarantees 100% exam success.",
    ]

    selected   = st.selectbox("Quick examples:", examples)
    default    = selected if selected != "Select an example..." else st.session_state.last_text
    text_input = st.text_area(
        "Paste any news claim, headline, or article text:",
        value=default, height=130,
        placeholder="Type or paste any news text here to analyze..."
    )

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze Now", use_container_width=True, type="primary")
    with col_btn2:
        clear_btn   = st.button("🗑️ Clear", use_container_width=True)
    with col_btn3:
        report_btn  = st.button("📄 PDF Report", use_container_width=True)

    if clear_btn:
        st.session_state.last_text = ""
        st.session_state.last_pred = None
        st.session_state.last_risk = None
        st.session_state.last_shap = []
        st.session_state.last_lime = []
        st.session_state.last_fkws = []
        st.session_state.last_rkws = []
        st.rerun()

    if analyze_btn and text_input.strip():
        progress = st.progress(0, text="Initializing...")
        for i in range(25):
            time.sleep(0.01)
            progress.progress(i, text="Running RoBERTa + keyword analysis...")

        pred     = predict(text_input, model, tokenizer)
        risk     = classify_risk(pred["prob_fake"])
        fake_kws = pred.get("fake_keywords_found", [])
        real_kws = pred.get("real_keywords_found", [])

        for i in range(25, 50):
            time.sleep(0.01)
            progress.progress(i, text="Computing risk level...")

        shap_data, lime_data = [], []
        if run_shap or run_explain:
            progress.progress(55, text="Computing SHAP values...")
            shap_data = compute_shap(text_input, model, tokenizer)

        if run_lime or run_explain:
            progress.progress(80, text="Computing LIME values...")
            lime_data = compute_lime(text_input, model, tokenizer)

        progress.progress(90, text="Verifying claim with live web search...")
        rt_result = verify_realtime(text_input)
        pred      = blend_verdicts(pred, rt_result, text=text_input)
        from claim_heuristics import apply_heuristic_to_prediction
        pred      = apply_heuristic_to_prediction(pred, text_input)
        risk      = classify_risk(pred["prob_fake"])
        fake_kws  = pred.get("fake_keywords_found", fake_kws)
        real_kws  = pred.get("real_keywords_found", real_kws)

        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

        st.session_state.last_text = text_input
        st.session_state.last_pred = pred
        st.session_state.last_risk = risk
        st.session_state.last_shap = shap_data
        st.session_state.last_lime = lime_data
        st.session_state.last_fkws = fake_kws
        st.session_state.last_rkws = real_kws

        already_saved = any(h["text"] == text_input[:80] + "..." for h in st.session_state.history)
        if not already_saved:
            st.session_state.history.append({
                "time":       datetime.now().strftime("%H:%M:%S"),
                "text":       text_input[:80] + "...",
                "verdict":    pred["label"],
                "risk":       risk["risk_level"],
                "prob_fake":  pred["prob_fake"],
                "confidence": risk["confidence"],
                "source":     "Single Analysis",
            })

    if st.session_state.last_pred is not None:
        pred      = st.session_state.last_pred
        risk      = st.session_state.last_risk
        shap_data = st.session_state.last_shap
        lime_data = st.session_state.last_lime
        fake_kws  = st.session_state.last_fkws
        real_kws  = st.session_state.last_rkws

        render_analysis_results(
            st.session_state.last_text, pred, risk, shap_data, lime_data,
            fake_kws, real_kws, run_shap, run_lime, run_semantic, run_explain,
        )

        if report_btn:
            with st.spinner("Generating PDF report..."):
                try:
                    from src.reporting.report_generator import (
                        save_shap_chart, save_lime_chart,
                        save_risk_bar, MisinformationReport
                    )
                    os.makedirs("reports", exist_ok=True)
                    shap_path = "reports/_tmp_shap.png"
                    lime_path = "reports/_tmp_lime.png"
                    risk_path = "reports/_tmp_risk.png"
                    save_shap_chart(shap_data, shap_path)
                    save_lime_chart(lime_data, lime_path)
                    save_risk_bar(pred["prob_fake"], risk_path)
                    cmap = {"LOW":(39,174,96),"MEDIUM":(180,150,0),"HIGH":(211,84,0),"CRITICAL":(192,57,43)}
                    vc   = cmap[risk["risk_level"]]
                    pdf = MisinformationReport()
                    pdf.add_page()
                    pdf.verdict_box(pred["label"], risk["risk_level"], vc)
                    pdf.section_title("1. Input Text")
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_fill_color(245, 245, 245)
                    pdf.multi_cell(0, 7, _safe(st.session_state.last_text), fill=True)
                    pdf.ln(3)
                    pdf.section_title("2. Classification Results")
                    pdf.key_value("Model",      "RoBERTa (roberta-base)")
                    pdf.key_value("Verdict",    pred["label"], (192,57,43) if pred["label"]=="Fake" else (39,174,96))
                    pdf.key_value("Risk Level", risk["risk_level"], vc)
                    pdf.key_value("Confidence", risk["confidence"])
                    pdf.key_value("Prob Real",  f"{pred['prob_real']:.4f}")
                    pdf.key_value("Prob Fake",  f"{pred['prob_fake']:.4f}")
                    pdf.key_value("Timestamp",  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    if pred.get("realtime_verdict"):
                        pdf.key_value("RT Verdict",     pred["realtime_verdict"])
                        pdf.key_value("RT Confidence",  f"{pred['realtime_confidence']:.0%}")
                        pdf.key_value("RT Explanation", pred.get("realtime_explanation", ""))
                    pdf.ln(3)
                    pdf.section_title("3. Risk Position")
                    pdf.insert_image(risk_path, w=170)
                    pdf.section_title("4. SHAP Token Contributions")
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(80, 80, 80)
                    pdf.multi_cell(0, 6, _safe("SHAP shows how each token contributed. Red = toward Fake, Green = toward Real."))
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(2)
                    pdf.insert_image(shap_path, w=170)
                    top_shap = sorted(shap_data, key=lambda x: abs(x[1]), reverse=True)[:6]
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_fill_color(200, 200, 200)
                    pdf.cell(70, 7, "Token",      fill=True, border=1, new_x="RIGHT", new_y="TOP")
                    pdf.cell(40, 7, "SHAP Value", fill=True, border=1, new_x="RIGHT", new_y="TOP")
                    pdf.cell(50, 7, "Direction",  fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "", 9)
                    for token, val in top_shap:
                        pdf.cell(70, 6, _safe(token), border=1, new_x="RIGHT", new_y="TOP")
                        pdf.cell(40, 6, f"{val:.4f}", border=1, new_x="RIGHT", new_y="TOP")
                        pdf.cell(50, 6, ">> Fake" if val > 0 else ">> Real", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(4)
                    pdf.section_title("5. LIME Word Contributions")
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(80, 80, 80)
                    pdf.multi_cell(0, 6, _safe("LIME perturbs input locally. Red = toward Fake, Green = toward Real."))
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(2)
                    pdf.insert_image(lime_path, w=170)
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_fill_color(200, 200, 200)
                    pdf.cell(70, 7, "Word",      fill=True, border=1, new_x="RIGHT", new_y="TOP")
                    pdf.cell(40, 7, "Weight",    fill=True, border=1, new_x="RIGHT", new_y="TOP")
                    pdf.cell(50, 7, "Direction", fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "", 9)
                    for word, weight in lime_data[:6]:
                        pdf.cell(70, 6, _safe(word), border=1, new_x="RIGHT", new_y="TOP")
                        pdf.cell(40, 6, f"{weight:.4f}", border=1, new_x="RIGHT", new_y="TOP")
                        pdf.cell(50, 6, ">> Fake" if weight > 0 else ">> Real", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(4)
                    pdf.section_title("6. Summary & Recommendation")
                    pdf.set_font("Helvetica", "", 10)
                    summaries = {
                        "LOW":      "Low misinformation risk. Content appears credible.",
                        "MEDIUM":   "Moderate risk. Manual review recommended.",
                        "HIGH":     "High risk. Do not share without verification.",
                        "CRITICAL": "Critical risk. Likely misinformation. Flag immediately.",
                    }
                    pdf.multi_cell(0, 7, _safe(summaries[risk["risk_level"]]))
                    path = f"reports/report_{datetime.now().strftime('%H%M%S')}.pdf"
                    pdf.output(path)
                    for p in [shap_path, lime_path, risk_path]:
                        if os.path.exists(p): os.remove(p)
                    with open(path, "rb") as f:
                        st.download_button("⬇️ Download PDF Report", f,
                                           file_name=os.path.basename(path),
                                           mime="application/pdf",
                                           use_container_width=True)
                    st.success(f"PDF saved to {path}")
                except Exception as e:
                    st.error(f"Report error: {str(e)}")

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")


# ════════════════════════════════════════════════════════
# TAB 2 — Batch
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Batch File Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aabbcc'>Upload CSV, TSV, TXT, or JSON files. "
        "Each line in a .txt file counts as one claim. For spreadsheets, pick the text column.</p>",
        unsafe_allow_html=True,
    )

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        <div style='background:#1e2a3a;border:1px solid #2d4a6e;border-radius:8px;padding:14px'>
            <p style='color:#00d4ff;font-weight:700;margin:0 0 8px'>Supported formats</p>
            <p style='color:#ddeeff;margin:4px 0;font-size:0.9rem'>
            <code style='color:#00d4ff'>.csv</code> &nbsp;
            <code style='color:#00d4ff'>.tsv</code> &nbsp;
            <code style='color:#00d4ff'>.txt</code> &nbsp;
            <code style='color:#00d4ff'>.json</code> &nbsp;
            <code style='color:#00d4ff'>.jsonl</code></p>
            <p style='color:#8899aa;font-size:0.85rem;margin:8px 0 0 0'>
            .txt = one claim per line</p>
        </div>""", unsafe_allow_html=True)
    with col_i2:
        st.markdown("""
        <div style='background:#1e2a3a;border:1px solid #2d4a6e;border-radius:8px;padding:14px'>
            <p style='color:#00d4ff;font-weight:700;margin:0 0 8px'>Notes:</p>
            <p style='color:#ddeeff;margin:4px 0;font-size:0.9rem'>✅ &nbsp;Works on any dataset</p>
            <p style='color:#ddeeff;margin:4px 0;font-size:0.9rem'>✅ &nbsp;Extra columns (labels, ids) kept in output</p>
            <p style='color:#ddeeff;margin:4px 0;font-size:0.9rem'>✅ &nbsp;Use slider to limit rows on large files</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose file (CSV, TXT, TSV, JSON, …)",
        type=BATCH_FILE_TYPES,
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            df, file_mode = read_uploaded_batch_file(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df, file_mode = None, None

        if df is not None and len(df.columns) == 0:
            st.error("File has no readable content.")
            df = None

        if df is not None:
            mode_label = "plain text (one line = one claim)" if file_mode == "lines" else "tabular data"
            st.success(
                f"Loaded **{len(df):,}** rows from `{uploaded.name}` ({mode_label})"
                + (f", **{len(df.columns)}** columns" if file_mode == "tabular" else "")
            )
            st.dataframe(df.head(3), use_container_width=True)

            if file_mode == "lines":
                text_col = "text"
                st.info("Plain text file: each non-empty line is analyzed as one claim.")
            else:
                text_candidates = detect_text_columns(df)
                text_col = st.selectbox(
                    "Column to analyze (text / claims / headlines):",
                    text_candidates,
                    index=0,
                    help="We auto-pick the most likely text column. Change this if predictions look wrong.",
                )

            default_rows = min(100, len(df))
            max_rows = st.slider(
                "Rows to analyze",
                min_value=1,
                max_value=len(df),
                value=default_rows,
                help=f"Large files ({len(df):,} rows): start with fewer rows, then increase.",
            )
            if len(df) > 500 and max_rows > 500:
                st.warning(f"Analyzing {max_rows:,} rows may take a long time. Consider starting with 100–500 rows.")

            df_run = df.head(max_rows).reset_index(drop=True)
            preview = df_run[text_col].dropna().astype(str).str.strip()
            preview = preview[preview != ""]
            if len(preview) == 0:
                st.error(f"Column `{text_col}` has no non-empty text values.")
            elif st.button("▶️ Run Batch Analysis", type="primary", use_container_width=True):
                rows   = []
                bar    = st.progress(0, text="Starting...")
                status = st.empty()
                total  = len(df_run)
                done   = 0

                for idx, row in df_run.iterrows():
                    text = str(row[text_col]).strip()
                    if not text or text.lower() == "nan":
                        continue
                    status.markdown(
                        f"<p style='color:#aabbcc'>Row {done + 1}/{total}: "
                        f"<code>{text[:60]}{'...' if len(text) > 60 else ''}</code></p>",
                        unsafe_allow_html=True,
                    )
                    p = predict(text, model, tokenizer)
                    r = classify_risk(p["prob_fake"])
                    out = {str(k): row[k] for k in df_run.columns}
                    out["analyzed_text"] = text[:200]
                    out["verdict"]       = p["label"]
                    out["prob_fake"]     = p["prob_fake"]
                    out["prob_real"]     = p["prob_real"]
                    out["risk_level"]    = f"{r['icon']} {r['risk_level']}"
                    out["confidence"]    = r["confidence"]
                    rows.append(out)
                    done += 1
                    bar.progress(done / total, text=f"Analyzed {done}/{total} rows")

                status.empty()
                bar.empty()

                if not rows:
                    st.error("No valid text rows found in the selected column.")
                else:
                    result_df = pd.DataFrame(rows)
                    st.session_state.batch_results = result_df

                    for _, row in result_df.iterrows():
                        snippet = str(row.get("analyzed_text", row.get(text_col, "")))[:80]
                        st.session_state.history.append({
                            "time":       datetime.now().strftime("%H:%M:%S"),
                            "text":       snippet + ("..." if len(snippet) >= 80 else ""),
                            "verdict":    row["verdict"],
                            "risk":       row["risk_level"],
                            "prob_fake":  row["prob_fake"],
                            "confidence": row["confidence"],
                            "source":     "Batch Analysis",
                        })
                    st.success(f"Analyzed {len(result_df)} rows. Results saved to History tab!")

    def show_batch_results(result_df):
        st.markdown("---")
        total_r = len(result_df)
        fake_r  = (result_df["verdict"] == "Fake").sum()
        real_r  = total_r - fake_r
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",  total_r)
        m2.metric("Fake",   fake_r)
        m3.metric("Real",   real_r)
        m4.metric("Fake %", f"{fake_r/total_r:.1%}")
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#1e2a3a")
        ax.set_facecolor("#1e2a3a")
        rc = result_df["risk_level"].value_counts()
        ax.bar(rc.index, rc.values, color=["#27ae60","#f1c40f","#e67e22","#c0392b"][:len(rc)], edgecolor="none")
        ax.set_title("Risk Distribution", color="#00d4ff", fontweight="bold")
        ax.tick_params(colors="#aabbcc")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#2d4a6e")
        ax.spines["bottom"].set_color("#2d4a6e")
        for i, v in enumerate(rc.values):
            ax.text(i, v+0.1, str(v), ha="center", color="#aabbcc", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(result_df, use_container_width=True, height=300)
        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv_out, "batch_results.csv", "text/csv", use_container_width=True)

    if st.session_state.batch_results is not None and not uploaded:
        st.info("Showing last batch results. Upload a new file to run again.")
        show_batch_results(st.session_state.batch_results)
    elif uploaded and st.session_state.batch_results is not None:
        show_batch_results(st.session_state.batch_results)


# ════════════════════════════════════════════════════════
# TAB 3 — History
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Analysis History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center; color:#556677; padding:60px 0'>
            <div style='font-size:3rem'>📭</div>
            <div style='color:#aabbcc;margin-top:10px'>No analyses yet. Go to Analyze Text tab to get started.</div>
        </div>""", unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        sources = ["All"] + list(hist_df["source"].unique()) if "source" in hist_df.columns else ["All"]
        filter_src = st.selectbox("Filter by source:", sources)
        if filter_src != "All":
            hist_df = hist_df[hist_df["source"] == filter_src]

        total_h = len(hist_df)
        fake_h  = (hist_df["verdict"] == "Fake").sum()
        real_h  = total_h - fake_h
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Total", total_h)
        hm2.metric("Fake",  fake_h)
        hm3.metric("Real",  real_h)

        st.dataframe(hist_df, use_container_width=True, height=300)

        if len(hist_df) > 1:
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#1e2a3a")
            ax.set_facecolor("#1e2a3a")
            colors = ["#e74c3c" if v == "Fake" else "#27ae60" for v in hist_df["verdict"]]
            ax.bar(range(len(hist_df)), hist_df["prob_fake"], color=colors, edgecolor="none")
            ax.axhline(0.5, color="#556677", linestyle="--", linewidth=1)
            ax.set_xlabel("Analysis #", color="#8899aa")
            ax.set_ylabel("Fake Probability", color="#8899aa")
            ax.set_title("History: Fake Probability per Analysis", color="#00d4ff", fontweight="bold")
            ax.tick_params(colors="#aabbcc")
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#2d4a6e")
            ax.spines["bottom"].set_color("#2d4a6e")
            plt.tight_layout()
            st.pyplot(fig)

        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History", csv_hist, "history.csv", "text/csv", use_container_width=True) 