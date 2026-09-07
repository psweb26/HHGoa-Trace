"""Genuine Google Lens reverse-image search through SerpApi."""

from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image


class ReverseSearchError(RuntimeError):
    """Raised when SerpApi cannot return a usable, genuine image-match result."""


SERPAPI_IMAGE_ENDPOINT = "https://serpapi.com/image"
SERPAPI_SEARCH_ENDPOINT = "https://serpapi.com/search.json"
SOCIAL_DOMAINS = {
    "instagram.com": "Instagram",
    "facebook.com": "Facebook",
    "x.com": "X",
    "twitter.com": "X/Twitter",
    "tiktok.com": "TikTok",
    "linkedin.com": "LinkedIn",
    "youtube.com": "YouTube",
    "reddit.com": "Reddit",
    "pinterest.com": "Pinterest",
    "threads.net": "Threads",
    "youtu.be": "YouTube",
}
POST_KEYWORDS = {
    "post", "reel", "video", "watch", "short", "tweet", "status", "thread", "pin", "comment", "clip",
}
NON_POST_PATH_SEGMENTS = {
    "", "about", "accounts", "channel", "channels", "explore", "hashtag", "hashtags", "home", "login",
    "profile", "profiles", "search", "settings", "signin", "signup", "tags",
}


def search_by_image(image_path: str | Path, timeout_seconds: int = 60) -> dict[str, Any]:
    """Upload ``image_path`` and return a selected match from live Lens results.

    No URL, title, or result is supplied by this program: every selected field comes
    from the current SerpApi response.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key or api_key == "your_serpapi_key_here":
        raise ReverseSearchError("SERPAPI_KEY is missing. Add it to a local .env file.")

    path = Path(image_path)
    if not path.is_file():
        raise ReverseSearchError(f"Input image does not exist: {path}")

    upload_name, upload_bytes, mime_type = _prepare_upload(path)
    try:
        upload_response = requests.post(
            SERPAPI_IMAGE_ENDPOINT,
            data={"api_key": api_key},
            files={"image": (upload_name, upload_bytes, mime_type)},
            timeout=timeout_seconds,
        )
        upload_response.raise_for_status()
        upload_data = upload_response.json()
    except requests.RequestException as exc:
        raise ReverseSearchError(f"SerpApi image upload failed: {exc}") from exc
    except ValueError as exc:
        raise ReverseSearchError("SerpApi image upload returned invalid JSON.") from exc

    if upload_data.get("error"):
        raise ReverseSearchError(f"SerpApi image upload error: {upload_data['error']}")
    image_id = upload_data.get("image_id")
    if not image_id:
        raise ReverseSearchError("SerpApi did not return an image_id for the uploaded image.")

    # Run a live exact-match search first. Only if it cannot establish an
    # individual social post do we incur a second, live visual-match search.
    exact_results = _lens_search(api_key, image_id, "exact_matches", timeout_seconds)
    candidate = _select_result(exact_results)
    if candidate is None:
        visual_results = _lens_search(api_key, image_id, "visual_matches", timeout_seconds)
        candidate = _select_result(visual_results)
    if candidate is None:
        raise ReverseSearchError(
            "Google Lens returned no validated individual public social-media post "
            "in its exact/visual matches. "
            "No record was written to the blockchain."
        )
    candidate["query_image_id"] = image_id
    candidate["query_upload_sha256"] = hashlib.sha256(upload_bytes).hexdigest()
    return candidate


def _lens_search(api_key: str, image_id: str, search_type: str, timeout_seconds: int) -> dict[str, Any]:
    try:
        search_response = requests.get(
            SERPAPI_SEARCH_ENDPOINT,
            params={
                "engine": "google_lens",
                "image_id": image_id,
                "type": search_type,
                "hl": "en",
                "safe": "active",
                "api_key": api_key,
            },
            timeout=timeout_seconds,
        )
        search_response.raise_for_status()
        results = search_response.json()
    except requests.RequestException as exc:
        raise ReverseSearchError(f"Google Lens {search_type} search through SerpApi failed: {exc}") from exc
    except ValueError as exc:
        raise ReverseSearchError("Google Lens search returned invalid JSON.") from exc
    if results.get("error"):
        raise ReverseSearchError(f"Google Lens {search_type} search error: {results['error']}")
    return results


def _prepare_upload(path: Path) -> tuple[str, bytes, str]:
    """Create a supported upload under SerpApi's 500 KiB image-upload limit."""
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ReverseSearchError(f"Could not prepare image for SerpApi: {exc}") from exc

    # Re-encoding also makes BMP/TIFF/etc. compatible with the documented API.
    for max_dimension, quality in ((2048, 88), (1600, 82), (1200, 76), (900, 70), (640, 62)):
        candidate = image.copy()
        candidate.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
        content = buffer.getvalue()
        if len(content) <= 500 * 1024:
            return f"{path.stem}.jpg", content, "image/jpeg"
    raise ReverseSearchError("Image could not be compressed below SerpApi's 500 KiB upload limit.")


def _select_result(results: dict[str, Any]) -> dict[str, Any] | None:
    """Return only an API-returned candidate validated as an individual public post.

    Exact matches outrank visual matches. Inside a section, platform-specific URL
    proof, a thumbnail, reported similarity, and meaningful result text determine
    the order. A social domain alone is deliberately insufficient.
    """
    for field, match_type in (("exact_matches", "exact_match"), ("visual_matches", "visual_match")):
        items = results.get(field)
        if not isinstance(items, list):
            continue
        normalized = [
            candidate
            for item in items
            if isinstance(item, dict)
            for candidate in [_normalize_candidate(item, match_type, field)]
            if candidate is not None
        ]
        if normalized:
            return max(normalized, key=_candidate_priority)
    return None


def _normalize_candidate(item: dict[str, Any], match_type: str, result_section: str) -> dict[str, Any] | None:
    url = item.get("link") or item.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        return None
    hostname = urlparse(url).hostname or ""
    hostname = hostname.lower().removeprefix("www.")
    platform = next((name for domain, name in SOCIAL_DOMAINS.items() if hostname == domain or hostname.endswith(f".{domain}")), None)
    if platform is None:
        return None
    source = item.get("source")
    if not isinstance(source, str) or not source.strip():
        source = hostname
    title = item.get("title") if isinstance(item.get("title"), str) else ""
    snippet = item.get("snippet") if isinstance(item.get("snippet"), str) else ""
    thumbnail_url = item.get("thumbnail") if _is_http_url(item.get("thumbnail")) else ""
    matched_image_url = next((value for value in (item.get("image"), thumbnail_url, item.get("image_source")) if _is_http_url(value)), "")
    validation_method = _validate_post_url(platform, url)
    if validation_method is None and _metadata_supports_individual_post(platform, url, title, snippet):
        validation_method = "metadata_supported_url"
    if validation_method is None:
        return None
    return {
        "url": url,
        "platform": platform,
        "source": source,
        "title": title,
        "snippet": snippet,
        "match_type": match_type,
        "result_type": "social_media_post",
        "result_section": result_section,
        "post_url_validation": validation_method,
        "thumbnail_url": thumbnail_url,
        "matched_image_url": matched_image_url,
        "image_similarity": _extract_similarity(item),
    }


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, int, float, int]:
    """Priority for already-validated posts, preserving provider order on a tie."""
    validation_strength = 2 if candidate["post_url_validation"] == "platform_url_pattern" else 1
    thumbnail_present = int(bool(candidate["thumbnail_url"]))
    similarity = candidate["image_similarity"]
    similarity_score = similarity if isinstance(similarity, (int, float)) else -1.0
    text_quality = int(bool(candidate["title"].strip())) + int(bool(candidate["snippet"].strip()))
    return validation_strength, thumbnail_present, similarity_score, text_quality


def _validate_post_url(platform: str, url: str) -> str | None:
    """Recognize documented, content-specific URL shapes for supported platforms."""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    query = parsed.query
    lower_parts = [part.lower() for part in parts]

    if platform == "Instagram" and len(parts) >= 2 and lower_parts[0] in {"p", "reel", "tv"}:
        return "platform_url_pattern"
    if platform == "X/Twitter" and len(parts) >= 3 and lower_parts[1] == "status" and parts[2]:
        return "platform_url_pattern"
    if platform == "Facebook":
        if (len(parts) >= 3 and lower_parts[1] == "posts" and parts[2]) or (len(parts) >= 2 and lower_parts[0] == "posts" and parts[1]):
            return "platform_url_pattern"
        if lower_parts[:1] in (["reel"], ["videos"]) and len(parts) >= 2:
            return "platform_url_pattern"
        if lower_parts[:1] == ["watch"] and (len(parts) >= 2 or re.search(r"(?:^|&)v=[^&]+", query)):
            return "platform_url_pattern"
        if parsed.path.lower().endswith(("permalink.php", "photo.php", "video.php")) and re.search(r"(?:^|&)(?:story_fbid|fbid|v)=", query):
            return "platform_url_pattern"
    if platform == "LinkedIn" and ((len(parts) >= 2 and lower_parts[0] == "posts") or lower_parts[:2] == ["feed", "update"]):
        return "platform_url_pattern"
    if platform == "TikTok" and ((len(parts) >= 3 and parts[0].startswith("@") and lower_parts[1] == "video") or (len(parts) >= 2 and lower_parts[0] == "video")):
        return "platform_url_pattern"
    if platform == "YouTube":
        if parsed.netloc.lower().removeprefix("www.") == "youtu.be" and len(parts) >= 1 and parts[0]:
            return "platform_url_pattern"
        if (parsed.path == "/watch" and re.search(r"(?:^|&)v=[^&]+", query)) or (len(parts) >= 2 and lower_parts[0] in {"shorts", "live", "embed"}):
            return "platform_url_pattern"
    if platform == "Threads" and len(parts) >= 3 and parts[0].startswith("@") and lower_parts[1] == "post" and parts[2]:
        return "platform_url_pattern"
    if platform == "Reddit" and len(parts) >= 4 and lower_parts[0] == "r" and lower_parts[2] == "comments":
        return "platform_url_pattern"
    if platform == "Pinterest" and len(parts) >= 2 and lower_parts[0] == "pin" and parts[1]:
        return "platform_url_pattern"
    return None


def _metadata_supports_individual_post(platform: str, url: str, title: str, snippet: str) -> bool:
    """Conservative fallback for valid provider URL variants not covered above."""
    parsed = urlparse(url)
    parts = [part.lower() for part in parsed.path.split("/") if part]
    if not parts or any(part in NON_POST_PATH_SEGMENTS for part in parts):
        return False
    # A lone username/profile is not enough evidence, even when its result text is rich.
    if len(parts) == 1 and not parsed.query:
        return False
    metadata_words = set(re.findall(r"[a-z]+", f"{title} {snippet}".lower()))
    has_content_language = bool(metadata_words & POST_KEYWORDS)
    has_specific_identifier = any(len(part) >= 6 and any(character.isdigit() for character in part) for part in parts)
    return has_content_language and (has_specific_identifier or bool(parsed.query)) and platform in SOCIAL_DOMAINS.values()


def _extract_similarity(item: dict[str, Any]) -> float | str | None:
    """Preserve an independently returned similarity field when Lens exposes one."""
    for key in ("image_similarity", "similarity", "score", "confidence"):
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str) and value.strip():
            numeric = value.strip().rstrip("%")
            try:
                return float(numeric)
            except ValueError:
                return value.strip()
    return None


def _is_http_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("https://", "http://"))
