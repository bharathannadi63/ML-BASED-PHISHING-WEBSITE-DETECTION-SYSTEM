from __future__ import annotations

import re
from urllib.parse import urlparse


def extract_features(url: str) -> list[int]:
    """
    Extract a fixed-length feature vector from a URL.

    IMPORTANT: Keep the feature count/order stable once a model is trained.
    Current features (15):
    1) length
    2) starts_with_https
    3) contains_ip
    4) contains_at
    5) dot_count
    6) contains_hyphen
    7) contains_suspicious_keywords
    8) has_query_params
    9) subdomain_count
    10) has_port
    11) url_entropy
    12) has_file_extension
    13) domain_length
    14) contains_double_slash
    15) special_char_count
    """
    if url is None:
        url = ""

    url = str(url).strip()

    # Ensure urlparse has a netloc if user enters "example.com" without scheme
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.netloc or ""
        query = parsed.query or ""
    except (ValueError, Exception):
        # Handle invalid URLs (e.g., malformed IPv6) by using empty host
        host = ""
        query = ""
    full = url  # original string for simple checks

    features: list[int] = []

    # 1) length (original input length after strip)
    features.append(len(url))

    # 2) starts_with_https (case-insensitive) - legitimate is typically https
    features.append(1 if url.lower().startswith("https") else 0)

    # 3) contains_ip (host-based detection) - use simpler check
    has_ip = 1 if (re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', host) is not None) else 0
    features.append(has_ip)

    # 4) contains_at
    features.append(1 if "@" in full else 0)

    # 5) dot_count (host-only is more meaningful than full URL)
    features.append(host.count("."))

    # 6) contains_hyphen (host-only)
    features.append(1 if "-" in host else 0)

    # 7) contains_suspicious_keywords
    suspicious_keywords = ['login', 'signin', 'verify', 'account', 'update', 'confirm', 
                          'password', 'secure', 'bank', 'paypal', 'amazon', 'apple']
    has_suspicious = 1 if any(keyword in url.lower() for keyword in suspicious_keywords) else 0
    features.append(has_suspicious)

    # 8) has_query_params
    features.append(1 if query else 0)

    # 9) subdomain_count
    subdomain_count = host.count(".") - 1 if host.count(".") > 0 else 0
    features.append(max(0, min(subdomain_count, 3)))  # cap at 3

    # 10) has_port (contains colon after domain)
    features.append(1 if ":" in host else 0)

    # 11) url_entropy (simplified - just count unique chars instead of calculating entropy)
    # Phishing often has many different characters
    unique_chars = len(set(host))
    features.append(min(unique_chars, 15))  # cap at 15

    # 12) has_file_extension (legitimate sites often don't have obvious file extensions in domain)
    common_extensions = ['.php', '.asp', '.jsp', '.exe', '.jar']
    has_ext = 1 if any(ext in url.lower() for ext in common_extensions) else 0
    features.append(has_ext)

    # 13) domain_length
    domain_length = len(host.split('.')[-2]) if host.count('.') > 0 else len(host)
    features.append(min(domain_length, 20))  # cap at 20

    # 14) contains_double_slash (beyond protocol://)
    double_slash_count = full.count('//') - 1  # subtract protocol
    features.append(1 if double_slash_count > 0 else 0)

    # 15) special_char_count in URL (phishing often has many special chars)
    # Count non-alphanumeric, non-slash, non-dot, non-hyphen characters
    special_chars = sum(1 for c in url if not (c.isalnum() or c in ':/.-?#&='))
    features.append(min(special_chars, 5))  # cap at 5

    return features
