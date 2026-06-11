# Project Diary: oss-pulse

This document records the journey of building oss-pulse: the limitations
we hit, the pivots we made, and what we learned along the way. Written
as source material for the final article.

---

## Day 1 (2026-06-10): From Idea to First Data

### The Plan

Analyze 10 years of Pull Request activity across the top 200 open-source
repos on GitHub. Use BigQuery to query the public GH Archive dataset,
then run time-series decomposition, forecasting, survival analysis, and
build a composite health index.

### Finding the Right Repos

First query to BigQuery: "top 200 repos by PR contributor count in 2025."

The results were disappointing. The list was dominated by educational repos:
`first-contributions`, IBM developer courses, `dio-lab-open-source`,
`Spoon-Knife`. These aren't software projects; they're onboarding exercises
where every student opens a PR as homework.

We added exclusion filters and re-ran. The list now had real projects:
NixOS, PyTorch, LLVM, Godot, Rust, Home Assistant. But this raised a
deeper question for the study itself: **what counts as a "top" open-source
project?** Stars? Contributors? PR volume? Each metric tells a different
story. We went with contributor count but noted this as a methodological
choice that affects results.

### BigQuery: The 1TB Wall

GH Archive is a public dataset on BigQuery. Free tier: 1TB of data scanned
per month. We thought that was plenty.

It wasn't. Each daily GH Archive table is roughly 5GB. Scanning one full
year (365 tables) = ~1.8TB. Our repo discovery query plus a single year
(2016) consumed the entire monthly quota. The rest of the decade would need
~18TB total.

The confusion: 18TB is not 18TB of data to download. BigQuery scans entire
tables internally even when your query filters to a fraction of the rows.
The actual extracted data for 2016 was 9MB on disk. Five orders of magnitude
difference between "data scanned" and "data returned."

Options considered:
- Pay ~$90 for 18TB of scans
- Wait for monthly quota reset and extract one year at a time (10 months)
- Switch to the GitHub API
- Download raw GH Archive JSON files and process locally

### Pivot: GitHub GraphQL API

We chose the GitHub API. Free, no scan quota, and the GraphQL endpoint
lets us pull exactly the fields we need in a single request per page of
100 PRs. As a bonus, we get first-review data in the same query (BigQuery
would have needed a separate pass for `PullRequestReviewEvent`).

The tradeoff: much slower. BigQuery can scan a year of data in seconds.
The GitHub API paginates through PRs one page at a time, with rate limits.

### First API Test: 502 Bad Gateway

Tested on `godotengine/godot`. After ~40 pages (3,035 PRs), GitHub
returned a 502 Bad Gateway. No warning, no error message, just a dead
connection.

This isn't documented anywhere. GitHub's own API docs discuss rate limits
(5,000 requests/hour) but not stability under sustained pagination. The
rate limit was never the bottleneck. We were using maybe 2,000 requests/hour.
The problem was that GitHub's GraphQL backend doesn't like 400+ sequential
requests to the same repository without a break.

Added exponential backoff (2s, 4s, 8s, 16s, 32s retries). Second test on
`modelcontextprotocol/servers` (3,110 PRs): completed cleanly in 3.7 minutes.

### The Overnight Run That Barely Worked

Estimated 3-4 hours for 200 repos. Left it running overnight.

Morning result: 4 repos completed in 9 hours. 14 attempted, 9 fatal errors.
51 retry attempts on 502 errors. The extraction had been hammering the
mega-repos (NixOS with 68k PRs, winget-pkgs with 104k) and drowning in 502s.

The estimate was wrong because we tested on a 3k-PR repo and extrapolated
linearly. But the distribution is extremely skewed: the top 10 repos have
10-100x more PRs than the median. And those are precisely the repos that
trigger API instability.

**Industry insight**: This is exactly why GH Archive exists. GitHub's API
was never designed for bulk historical extraction. GH Archive captures the
event stream in real-time and stores it for batch access. Our original
BigQuery approach was architecturally correct, just financially impractical
on the free tier.

### Real Data vs Pipeline: Two Incompatibilities

When the first repos finished downloading, we ran them through the pipeline.
Two things broke immediately:

1. **Null authors**: GitHub returns `null` for the author field when a user
   account has been deleted. Our bot detection function called `.lower()` on
   the username and crashed on `None`. ~1.5% of PRs in real data have deleted
   authors. Fix: treat null authors as "ghost" (GitHub's own convention).

2. **The "merged" state**: GitHub's GraphQL API returns `state: "MERGED"` as
   a distinct state. Our pipeline expected `state: "closed"` + `merged: "true"`
   (the GH Archive convention). Fix: normalize `state=merged` to
   `state=closed, merged=true` in the clean step.

Neither of these showed up in synthetic data or in the GH Archive extract,
because the synthetic generator doesn't model deleted users, and GH Archive
uses a different state representation. **The pipeline only becomes correct
when you run real data through it.** Synthetic tests verify mechanics;
real data verifies assumptions.

### The Fix: Small Repos First

Analyzed the repo list: 196 of 200 repos have fewer than 30,000 events.
Only 4 are "mega-repos" (winget-pkgs, nixpkgs, homebrew-core, homebrew-cask).

Changed strategy:
- Sort repos by size, smallest first
- Add 1-second delay between API pages to reduce 502s
- Defer the 4 mega-repos for later (BigQuery when quota resets, or date-range splitting)
- Per-repo parquet caching so the process can resume if interrupted

Relaunched. First repos coming in clean, zero errors, zero retries. The
delay means each repo takes slightly longer, but reliability is worth more
than speed when you're extracting a dataset you'll analyze for months.

**Lesson**: When extracting data at scale, start with the easy wins. 196
repos with clean data is more valuable than 4 repos with errors. Don't let
the hardest 2% block the other 98%.

### The nohup Gotcha

Small but classic: launched the extraction with `nohup python ...` and
watched the log file. Nothing appeared for 10 minutes. Thought the process
was stuck.

It wasn't. Python buffers stdout when writing to a file (non-TTY). The
data was accumulating in memory, not hitting disk. Fix: `python -u` for
unbuffered output. Lost 10 minutes to this.

---

## Technical Decisions

### Why Two Data Sources (BigQuery + GitHub API)

We ended up needing both:
- **BigQuery** for repo discovery (the top-repos query scans recent data
  efficiently, fits in free tier)
- **GitHub API** for historical PR extraction (no scan quota, resume-safe,
  richer per-request data including reviews)

This hybrid wasn't planned. It emerged from hitting BigQuery's quota limit.
In retrospect, it's actually a better architecture: BigQuery for the index,
API for the detail.

### Why Parquet Over CSV

Parquet is 5-10x smaller for this data shape, preserves types (timestamps
and ints don't need re-parsing), and reads 10-50x faster with pyarrow.
For a dataset that gets read hundreds of times during analysis, this adds up.

### GitHub API State Naming

GitHub's GraphQL API returns PR states as `OPEN`/`CLOSED`/`MERGED` (uppercase).
GH Archive uses lowercase. Our transform pipeline normalizes both, but this
is the kind of thing that breaks silently if you merge data from both sources
without a cleaning step.

---

## Timeline

| Time | Event |
|------|-------|
| 20:00 | Project kickoff |
| 20:30 | Pipeline code complete (extract, transform, analyze, visualize) |
| 21:00 | 134 tests, 85% coverage, CI green |
| 22:00 | BigQuery works! First query returns 220k events for a single day |
| 22:05 | BigQuery quota exhausted. Only 1 of 10 years extracted. |
| 22:15 | Pivoted to GitHub GraphQL API |
| 22:30 | First API test: 502 after 3k PRs. Added retry logic. |
| 22:40 | Second test: 3,110 PRs clean. Launched full extraction. |
| 23:00 | Discovered nohup stdout buffering. Relaunched with -u flag. |
| 08:00 | Morning check: 4/200 repos in 9 hours. 51 retries, 9 fatal errors. |
| 09:30 | Rewrite: small repos first, 1s delay, defer mega-repos. Relaunched. |
| 09:45 | New extraction running clean. Zero errors. |
| --- | **Day 2 (2026-06-11)** |
| 10:00 | Discovered extraction data was in wrong directory (sandbox, not oss-pulse). Venv too. |
| 10:15 | Lost all cached repo parquets. Also lost BigQuery top_repos (replaced by synthetic). |
| 10:20 | Recreated venv inside oss-pulse. Re-discovered top 200 repos via GitHub Search API (by stars). |
| 10:25 | Relaunched extraction. Small repos completing in seconds. Many awesome-lists return 0 PRs (expected). |
