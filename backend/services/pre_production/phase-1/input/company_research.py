"""
Pre-Processing Step 1 — Company Research Extraction  (v2)

Five-stage pipeline:
  1. Discovery   — sitemap (recursive) + homepage nav links + Shopify JSON API
  2. Prioritise  — score URLs into 4 tiers, filter noise
  3. Crawl       — Playwright-rendered fetch per page
  4. Errors      — 429 retry, 403 alt-UA, timeout, empty — never silent
  5. Assemble    — delimited output, 150 k-word cap, low-content warning

Writes to MongoDB:
  pipeline   — execution log: pages_attempted, statuses, summary
  strategy   — shared context: company_research.raw_text
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
load_dotenv()

import httpx
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from motor.motor_asyncio import AsyncIOMotorDatabase

# ---------------------------------------------------------------------------
# Logger  (format set once in main.py via basicConfig; child loggers inherit)
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.company_research")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PAGES             = 40
MAX_WORDS             = 150_000
LOW_CONTENT_THRESHOLD = 2_000
MAX_BLOG_ARTICLES     = 3

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")

SITEMAP_PATHS = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap/",
    "/page-sitemap.xml",
]

SKIP_PATTERNS = [
    "/cart", "/checkout", "/account", "/login",
    "/register", "/password", "/cdn/", "/assets/", "/.well-known/",
]

PRIORITY_1_KEYWORDS = [
    "about", "story", "our-story", "who-we-are", "founders", "mission",
    "vision", "values", "philosophy", "why-us", "why-", "ingredients",
    "technology", "process", "how-we-make", "certifications",
    "sustainability", "giving-back", "impact", "ethics",
]
PRIORITY_2_KEYWORDS = [
    "collections", "products", "face", "hair", "body", "skincare",
    "haircare", "how-it-works", "routine", "regime", "solutions", "benefits",
]
PRIORITY_3_KEYWORDS = [
    "testimonials", "reviews", "faq", "questions", "results", "before-after",
]
PRIORITY_4_KEYWORDS = [
    "blog", "articles", "editorial",
]

USER_AGENTS = [
    # Primary — Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Fallback — Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def _browser_cfg(ua: str = USER_AGENTS[0]) -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
        user_agent=ua,
    )


def _crawl_cfg() -> CrawlerRunConfig:
    return CrawlerRunConfig(
        page_timeout=15_000,
        excluded_tags=["nav", "footer", "aside", "header", "script", "style", "form"],
        remove_overlay_elements=True,
        exclude_external_links=True,
        word_count_threshold=10,
        magic=True,
    )


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------

def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _normalise(url: str) -> str:
    """Strip fragment and trailing slash."""
    p = urlparse(url)
    return p._replace(fragment="").geturl().rstrip("/")


def _same_origin(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def _should_skip(url: str) -> bool:
    lower = url.lower()
    return any(pat in lower for pat in SKIP_PATTERNS)


def _assign_priority(url: str) -> int | None:
    if _should_skip(url):
        return None
    lower = url.lower()
    for kw in PRIORITY_1_KEYWORDS:
        if kw in lower:
            return 1
    for kw in PRIORITY_2_KEYWORDS:
        if kw in lower:
            return 2
    for kw in PRIORITY_3_KEYWORDS:
        if kw in lower:
            return 3
    for kw in PRIORITY_4_KEYWORDS:
        if kw in lower:
            return 4
    return 2  # default: treat unknown pages as product-tier


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _extract_markdown(result) -> str | None:
    md = result.markdown
    if md is None:
        return None
    if hasattr(md, "raw_markdown"):
        return md.raw_markdown or None
    return str(md) or None


# ---------------------------------------------------------------------------
# STAGE 1 — Discovery helpers
# ---------------------------------------------------------------------------

async def _fetch_text(
    client: httpx.AsyncClient, url: str, timeout: float = 10.0
) -> str | None:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


async def _parse_sitemap(
    client: httpx.AsyncClient,
    url: str,
    visited_sitemaps: set[str],
    page_urls: set[str],
) -> None:
    """Recursively parse sitemap / sitemap-index, collecting all page URLs."""
    norm = _normalise(url)
    if norm in visited_sitemaps:
        logger.debug("Sitemap already visited, skipping: %s", url)
        return
    visited_sitemaps.add(norm)

    text = await _fetch_text(client, url)
    if not text:
        logger.warning("Child sitemap failed (404 or timeout)  |  url=%s", url)
        return

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        logger.warning("Sitemap XML parse error  |  url=%s", url)
        return

    # Strip namespace from tag for reliable comparison
    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if tag == "sitemapindex":
        child_locs = [
            loc.text.strip()
            for loc in root.findall(f"{{{SITEMAP_NS}}}sitemap/{{{SITEMAP_NS}}}loc")
            if loc.text
        ]
        logger.info("Sitemap index  |  url=%s  children=%d", url, len(child_locs))
        for child_url in child_locs:
            await _parse_sitemap(client, child_url, visited_sitemaps, page_urls)

    elif tag == "urlset":
        locs = [
            loc.text.strip()
            for loc in root.findall(f"{{{SITEMAP_NS}}}url/{{{SITEMAP_NS}}}loc")
            if loc.text
        ]
        logger.info("Sitemap urlset  |  url=%s  pages=%d", url, len(locs))
        page_urls.update(locs)


async def _discover_sitemap(base: str) -> set[str]:
    visited: set[str] = set()
    page_urls: set[str] = set()
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENTS[0]}, follow_redirects=True
    ) as client:
        for path in SITEMAP_PATHS:
            url = base.rstrip("/") + path
            text = await _fetch_text(client, url)
            if text:
                logger.info("Sitemap found  |  %s", url)
                await _parse_sitemap(client, url, visited, page_urls)
                break
        else:
            logger.warning("No sitemap found  |  base=%s", base)
    return page_urls


async def _discover_sitemap_from_url(sitemap_url: str) -> set[str]:
    """
    Parse a user-provided sitemap URL.

    Fast path (httpx): works for standard XML sitemaps on unprotected servers.
    Fallback (Crawl4AI / Playwright): handles bot-protection (e.g. HTTP 418) and
    HTML sitemap pages by extracting all internal <a href> links directly.
    """
    page_urls: set[str] = set()
    logger.info("Using provided sitemap URL  |  %s", sitemap_url)

    # ── Fast path: try httpx ──────────────────────────────────────────────────
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENTS[0]}, follow_redirects=True
    ) as client:
        text = await _fetch_text(client, sitemap_url)

    if text:
        stripped = text.strip()
        is_xml = stripped.startswith("<") and (
            "urlset" in stripped[:500] or "sitemapindex" in stripped[:500]
        )
        if is_xml:
            visited: set[str] = set()
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENTS[0]}, follow_redirects=True
            ) as client:
                await _parse_sitemap(client, sitemap_url, visited, page_urls)
            logger.info("Sitemap (XML via httpx)  |  %d URLs", len(page_urls))
            return page_urls
        else:
            # HTML sitemap page — extract <a href> links directly
            soup = BeautifulSoup(text, "html.parser")
            base = _origin(sitemap_url)
            for a in soup.find_all("a", href=True):
                abs_url = urljoin(sitemap_url, a["href"])
                if _same_origin(abs_url, sitemap_url):
                    page_urls.add(_normalise(abs_url))
            if page_urls:
                logger.info("Sitemap (HTML via httpx)  |  %d URLs", len(page_urls))
                return page_urls

    # ── Fallback: Crawl4AI (bypasses bot protection / 418) ───────────────────
    logger.info(
        "httpx blocked or returned no content for sitemap URL — falling back to Crawl4AI  |  %s",
        sitemap_url,
    )
    cfg = CrawlerRunConfig(
        page_timeout=15_000,
        remove_overlay_elements=True,
        exclude_external_links=True,
        word_count_threshold=0,
    )
    try:
        async with AsyncWebCrawler(config=_browser_cfg(USER_AGENTS[0])) as crawler:
            result = await crawler.arun(sitemap_url, config=cfg)
            if result.success:
                # Prefer structured link extraction when available
                internal_links = (result.links or {}).get("internal", [])
                if internal_links:
                    for link in internal_links:
                        href = link.get("href", "")
                        if href and _same_origin(href, sitemap_url):
                            page_urls.add(_normalise(href))
                else:
                    # Fallback: parse BeautifulSoup over raw HTML
                    raw_html = getattr(result, "html", "") or ""
                    if raw_html:
                        soup = BeautifulSoup(raw_html, "html.parser")
                        for a in soup.find_all("a", href=True):
                            abs_url = urljoin(sitemap_url, a["href"])
                            if _same_origin(abs_url, sitemap_url):
                                page_urls.add(_normalise(abs_url))
    except Exception as exc:
        logger.warning(
            "Crawl4AI sitemap fetch failed  |  %s  error=%s", sitemap_url, exc
        )

    logger.info("Sitemap (Crawl4AI fallback)  |  %d URLs  |  %s", len(page_urls), sitemap_url)
    return page_urls


async def _discover_homepage_links(base: str) -> set[str]:
    """
    Parse internal links from the homepage for URL discovery.

    Fast path (httpx): parses <a href> links from nav/header/footer elements.
    Fallback (Crawl4AI): used when httpx is blocked (e.g. HTTP 418).
      Crawl4AI returns all internal links via result.links["internal"], which
      includes nav links and all other in-domain hrefs on the page.
    """
    urls: set[str] = set()

    # ── Fast path: httpx ──────────────────────────────────────────────────────
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENTS[0]}, follow_redirects=True
    ) as client:
        text = await _fetch_text(client, base.rstrip("/") + "/")

    if text:
        soup = BeautifulSoup(text, "html.parser")
        for container in soup.find_all(["nav", "header", "footer"]):
            for a in container.find_all("a", href=True):
                abs_url = urljoin(base, a["href"])
                if _same_origin(abs_url, base):
                    urls.add(_normalise(abs_url))
        if urls:
            return urls

    # ── Fallback: Crawl4AI (bypasses bot-protection / 418) ───────────────────
    logger.info(
        "httpx blocked for homepage — using Crawl4AI for link discovery  |  base=%s", base
    )
    cfg = CrawlerRunConfig(
        page_timeout=15_000,
        remove_overlay_elements=True,
        exclude_external_links=True,
        word_count_threshold=0,
        magic=True,
    )
    try:
        async with AsyncWebCrawler(config=_browser_cfg(USER_AGENTS[0])) as crawler:
            result = await crawler.arun(base.rstrip("/") + "/", config=cfg)
            if result.success:
                for link in (result.links or {}).get("internal", []):
                    href = link.get("href", "")
                    if href and _same_origin(href, base):
                        urls.add(_normalise(href))
                logger.info(
                    "Crawl4AI homepage discovery  |  %d internal links  |  base=%s",
                    len(urls), base,
                )
    except Exception as exc:
        logger.warning(
            "Crawl4AI homepage link discovery failed  |  base=%s  error=%s", base, exc
        )
    return urls


async def _discover_shopify(base: str, homepage_text: str | None) -> set[str]:
    """If Shopify detected, fetch pages / collections / blog articles via JSON API."""
    urls: set[str] = set()
    if not homepage_text or "cdn.shopify.com" not in homepage_text:
        return urls
    logger.info("Shopify detected  |  base=%s", base)
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENTS[0]}, follow_redirects=True
    ) as client:
        # Custom pages
        text = await _fetch_text(client, base.rstrip("/") + "/pages.json?limit=250")
        if text:
            try:
                for page in json.loads(text).get("pages", []):
                    urls.add(_normalise(base.rstrip("/") + "/pages/" + page["handle"]))
            except Exception:
                pass
        # Collections
        text = await _fetch_text(client, base.rstrip("/") + "/collections.json?limit=250")
        if text:
            try:
                for col in json.loads(text).get("collections", []):
                    urls.add(_normalise(base.rstrip("/") + "/collections/" + col["handle"]))
            except Exception:
                pass
        # Blogs → articles (top 3 blogs, 3 articles each)
        text = await _fetch_text(client, base.rstrip("/") + "/blogs.json?limit=250")
        if text:
            try:
                for blog in json.loads(text).get("blogs", [])[:3]:
                    art = await _fetch_text(
                        client,
                        base.rstrip("/") + f"/blogs/{blog['handle']}/articles.json?limit=10",
                    )
                    if art:
                        for article in json.loads(art).get("articles", [])[:MAX_BLOG_ARTICLES]:
                            urls.add(_normalise(
                                base.rstrip("/")
                                + f"/blogs/{blog['handle']}/{article['handle']}"
                            ))
            except Exception:
                pass
    return urls


async def _discover_all(company_url: str, sitemap_url: str | None = None) -> list[tuple[str, int]]:
    """
    Stage 1 + 2: discover all URLs then assign priorities.
    Returns sorted list of (url, priority) — priority 1 first.

    If sitemap_url is provided it is used directly; otherwise the standard
    4-path probe is run (existing behaviour, unchanged).
    """
    base = _origin(company_url)
    logger.info("Discovery starting  |  base=%s", base)

    # Fetch homepage once — used for Shopify detection and nav extraction
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENTS[0]}, follow_redirects=True
    ) as client:
        homepage_text = await _fetch_text(client, base.rstrip("/") + "/")

    sitemap_task = (
        _discover_sitemap_from_url(sitemap_url)
        if sitemap_url
        else _discover_sitemap(base)
    )

    sitemap_urls, homepage_urls, shopify_urls = await asyncio.gather(
        sitemap_task,
        _discover_homepage_links(base),
        _discover_shopify(base, homepage_text),
    )

    # Sitemap URLs are canonical — build a lowercase index so we can
    # detect case-variant duplicates from nav / Shopify discovery.
    sitemap_normalised = {_normalise(u) for u in sitemap_urls}
    sitemap_lower: dict[str, str] = {u.lower(): u for u in sitemap_normalised}

    all_urls: set[str] = {_normalise(base + "/")}
    all_urls.update(sitemap_normalised)

    # Only add nav / Shopify URLs when no sitemap URL matches case-insensitively.
    # This prevents crawling /pages/About-Us AND /pages/about-us separately.
    for u in homepage_urls | shopify_urls:
        norm = _normalise(u)
        if norm.lower() not in sitemap_lower:
            all_urls.add(norm)

    logger.info(
        "Discovery complete  |  sitemap=%d  nav=%d  shopify=%d  total_unique=%d",
        len(sitemap_urls), len(homepage_urls), len(shopify_urls), len(all_urls),
    )

    prioritised: list[tuple[str, int]] = []
    for url in all_urls:
        if not _same_origin(url, base):
            continue
        p = _assign_priority(url)
        if p is None:
            continue
        prioritised.append((url, p))

    prioritised.sort(key=lambda x: (x[1], x[0]))
    logger.info("Crawl queue built  |  %d URLs after skip filter", len(prioritised))
    return prioritised


# ---------------------------------------------------------------------------
# STAGE 3 + 4 — Fetch & extract with error handling
# ---------------------------------------------------------------------------

PageStatus = dict  # {url, priority, status, words?}


async def _crawl_one(
    crawler: AsyncWebCrawler,
    url: str,
    priority: int,
    alt_ua: bool = False,
) -> tuple[str | None, PageStatus]:
    """
    Crawl a single page. Returns (text | None, status_dict).
    Handles 429 retry, 403 alt-UA signal, timeout, empty.
    """
    cfg = _crawl_cfg()
    result = await crawler.arun(url, config=cfg)
    status_code = getattr(result, "status_code", 200) or 200

    # 429 — wait 5 s, retry once
    if status_code == 429:
        logger.warning("429 rate-limited  |  %s  — waiting 5 s then retrying", url)
        await asyncio.sleep(5)
        result = await crawler.arun(url, config=cfg)
        status_code = getattr(result, "status_code", 200) or 200
        if status_code == 429:
            return None, {"url": url, "priority": priority, "status": "rate_limited"}

    # 403 — signal caller to retry with alt UA
    if status_code == 403 and not alt_ua:
        return None, {"url": url, "priority": priority, "status": "needs_alt_ua"}

    if not result.success:
        if status_code == 403:
            s = "blocked"
        elif status_code in (0, 408):
            s = "timeout"
        else:
            s = f"http_{status_code}"
        logger.warning("  FAIL [P%d]  status=%s  |  %s", priority, s, url)
        return None, {"url": url, "priority": priority, "status": s}

    text = _extract_markdown(result)
    if not text or _word_count(text) < 10:
        logger.info("  EMPTY [P%d]  |  %s", priority, url)
        return None, {"url": url, "priority": priority, "status": "empty"}

    wc = _word_count(text)
    logger.info("  OK [P%d]  %d words  |  %s", priority, wc, url)
    return text, {"url": url, "priority": priority, "status": "success", "words": wc}


# ---------------------------------------------------------------------------
# STAGE 5 — Assemble output
# ---------------------------------------------------------------------------

def _assemble(pages: list[tuple[str, int, str]]) -> tuple[str, int]:
    """
    Concatenate pages in priority order with delimiters.
    Stops at a page boundary when MAX_WORDS would be exceeded.
    Returns (assembled_text, total_words).
    """
    parts: list[str] = []
    total_words = 0
    for url, priority, text in pages:
        wc = _word_count(text)
        if total_words + wc > MAX_WORDS:
            logger.info("150 k-word cap reached — stopping assembly")
            break
        parts.append(
            f"=== PAGE: {url} | PRIORITY: {priority} | WORDS: {wc} ===\n"
            f"{text}\n"
            f"=== END PAGE ==="
        )
        total_words += wc
    return "\n\n".join(parts), total_words


# ---------------------------------------------------------------------------
# Main crawl orchestrator
# ---------------------------------------------------------------------------

async def crawl_site(
    company_url: str,
    sitemap_url: str | None = None,
) -> tuple[int, str | None, str | None, dict]:
    """
    Full 5-stage crawl.
    Returns (pages_scraped, raw_text, error, summary).
    """
    base = _origin(company_url)

    # ── Abort immediately if homepage is unreachable ────────────────────────────
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENTS[0]}, follow_redirects=True
    ) as probe:
        try:
            r = await probe.get(base + "/", timeout=15)
            if r.status_code >= 500:
                logger.error("Site unreachable (HTTP %d)  |  %s", r.status_code, base)
                return 0, None, "site_unreachable", {"status": "site_unreachable"}
        except Exception as exc:
            logger.error("Site unreachable (exception)  |  %s  error=%s", base, exc)
            return 0, None, "site_unreachable", {
                "status": "site_unreachable", "error": str(exc)
            }

    # ── Stage 1 + 2: Discover & prioritise ────────────────────────────────────
    prioritised = await _discover_all(company_url, sitemap_url=sitemap_url)
    queue = prioritised[:MAX_PAGES]

    # ── Stage 3 + 4: Crawl each page ────────────────────────────────────────
    successes: list[tuple[str, int, str]] = []
    statuses:  list[PageStatus] = []
    visited:   set[str] = set()
    p4_count = 0

    async with (
        AsyncWebCrawler(config=_browser_cfg(USER_AGENTS[0])) as crawler_primary,
        AsyncWebCrawler(config=_browser_cfg(USER_AGENTS[1])) as crawler_alt,
    ):
        for url, priority in queue:
            norm = _normalise(url)
            if norm in visited:
                continue
            visited.add(norm)

            # Cap Priority-4 (blog) at MAX_BLOG_ARTICLES
            if priority == 4:
                if p4_count >= MAX_BLOG_ARTICLES:
                    continue
                p4_count += 1

            logger.info(
                "Crawling [%d/%d] P%d  |  %s",
                len(visited), len(queue), priority, url,
            )

            text, status = await _crawl_one(crawler_primary, url, priority)

            # Retry with alt UA on 403
            if status.get("status") == "needs_alt_ua":
                logger.info("Retrying with alt UA  |  %s", url)
                text, status = await _crawl_one(crawler_alt, url, priority, alt_ua=True)
                if status.get("status") == "needs_alt_ua":
                    status["status"] = "blocked"
                    text = None

            statuses.append(status)
            if text:
                successes.append((url, priority, text))

            await asyncio.sleep(0.5)

    # ── Stage 5: Assemble ────────────────────────────────────────────────────
    raw_text, total_words = _assemble(successes) if successes else ("", 0)

    summary = {
        "pages_attempted":  len(statuses),
        "pages_successful": sum(1 for s in statuses if s["status"] == "success"),
        "pages_empty":      sum(1 for s in statuses if s["status"] == "empty"),
        "pages_blocked":    sum(1 for s in statuses if s["status"] in ("blocked", "rate_limited")),
        "pages_timeout":    sum(1 for s in statuses if s["status"] == "timeout"),
        "total_words":      total_words,
        "empty_urls":       [s["url"] for s in statuses if s["status"] == "empty"],
        "blocked_urls":     [s["url"] for s in statuses if s["status"] in ("blocked", "rate_limited")],
    }

    if 0 < total_words < LOW_CONTENT_THRESHOLD:
        summary["warning"] = "low_content_site"
        summary["possible_reason"] = "JS-heavy or gated content"
        logger.warning(
            "Low content site  |  total_words=%d  threshold=%d",
            total_words, LOW_CONTENT_THRESHOLD,
        )

    logger.info(
        "Crawl summary  |  attempted=%d  success=%d  empty=%d  "
        "blocked=%d  timeout=%d  total_words=%d",
        summary["pages_attempted"], summary["pages_successful"],
        summary["pages_empty"],     summary["pages_blocked"],
        summary["pages_timeout"],   total_words,
    )

    error = None if summary["pages_successful"] > 0 else "no_pages_extracted"
    return summary["pages_successful"], raw_text or None, error, summary


# ---------------------------------------------------------------------------
# Entry point — called as a FastAPI BackgroundTask
# ---------------------------------------------------------------------------

async def run_company_research_extraction(
    project_id: str,
    company_url: str,
    db: AsyncIOMotorDatabase,
    sitemap_url: str | None = None,
    product_url: str | None = None,
) -> None:
    """
    Run the full crawl and persist results:
      pipeline  — lean log entry in pre_processing[] (no raw text)
      strategy  — company_research.raw_text
    Never raises — all errors are caught and stored.
    """
    logger.info(
        "[PRE-PROCESSING 1] Starting  |  project_id=%s  url=%s",
        project_id, company_url,
    )

    # Ensure indexes (no-op if already present)
    await db[PIPELINE_COLLECTION].create_index("project_id", unique=True)
    await db[STRATEGY_COLLECTION].create_index("project_id", unique=True)

    pages_scraped: int = 0
    raw_text:      str | None = None
    error:         str | None = None
    summary:       dict = {}

    try:
        pages_scraped, raw_text, error, summary = await crawl_site(company_url, sitemap_url=sitemap_url)
    except Exception as exc:
        error = str(exc)
        logger.error(
            "[PRE-PROCESSING 1] Unhandled error  |  project_id=%s  error=%s",
            project_id, exc,
        )

    now = datetime.now(timezone.utc)

    # ── pipeline: execution log ─────────────────────────────────────────────
    log_entry = {
        "step":          "company_research_extraction",
        "pages_scraped": pages_scraped,
        "scraped_at":    now,
        "error":         error,
        "summary":       summary,
    }
    await db[PIPELINE_COLLECTION].update_one(
        {"project_id": project_id},
        {
            "$setOnInsert": {"project_id": project_id, "agents": []},
            "$push": {"pre_processing": log_entry},
        },
        upsert=True,
    )
    logger.info(
        "[PRE-PROCESSING 1] pipeline log written  |  project_id=%s  pages=%d  error=%s",
        project_id, pages_scraped, error,
    )

    # ── strategy: raw_text + product_url for agents ──────────────────────────
    strategy_set = {"company_research.raw_text": raw_text}
    if product_url:
        strategy_set["product_url"] = product_url
    await db[STRATEGY_COLLECTION].update_one(
        {"project_id": project_id},
        {
            "$setOnInsert": {
                "project_id": project_id,
                "phase":      1,
                "created_at": now,
                "updated_at": now,
            },
            "$set": strategy_set,
        },
        upsert=True,
    )
    logger.info(
        "[PRE-PROCESSING 1] strategy written  |  project_id=%s  raw_text=%s",
        project_id, "present" if raw_text else "null",
    )
    logger.info("[PRE-PROCESSING 1] Done  |  project_id=%s", project_id)
