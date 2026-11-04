"""Seed data for the synthetic testbed repo (a fictional HTTP client called ``acme_http``).

Every entry is invented. ``query`` is how a user would *describe* the problem in their own words;
it is deliberately not a copy of the title, so lexical search has to earn its hits.

resolution:
  fixed    - closed by a merged PR whose body says "Fixes #N"  (state_reason: completed)
  answered - closed as completed with the answer in a comment, no PR
  wontfix  - closed as not planned, with the maintainer's reason in a comment
  open     - still open (a workaround may be in the comments)
"""

from __future__ import annotations

ISSUES: dict[str, dict] = {
    "pool": dict(
        title="Connection pool exhausted under load",
        body="With 64 worker threads sharing one client, calls start blocking forever. "
        "`pool_maxsize` defaults to 10 and there is no way to bound the wait.",
        comments=[
            "Reproduced. The pool hands out connections but never times out waiting for one."
        ],
        resolution="fixed",
        file="acme_http/pool.py",
        fix="Raise default pool size and add `pool_timeout` so waiters fail fast",
        query="requests hang when lots of threads hit the API at the same time and no sockets are left",
    ),
    "stream_timeout": dict(
        title="timeout= ignored when streaming a download",
        body="`client.get(url, stream=True, timeout=5)` still hangs for minutes when the server "
        "stops sending bytes mid-body.",
        comments=["The timeout only covered connect and the first byte, not subsequent reads."],
        resolution="fixed",
        file="acme_http/streaming.py",
        fix="Apply the read timeout to every chunk, not only the first",
        query="my download never gives up even though I set a time limit",
    ),
    "proxy_redirect": dict(
        title="Proxy-Authorization sent to the redirect target",
        body="When a response redirects to a different host, the proxy credentials are forwarded "
        "along with the request. Security-sensitive.",
        comments=["Confirmed, headers must be stripped when the host changes."],
        resolution="fixed",
        file="acme_http/auth.py",
        fix="Strip Proxy-Authorization when a redirect crosses hosts",
        query="my proxy password shows up in requests to another site after the server redirects me",
    ),
    "post_retry": dict(
        title="Failed POST is sent twice",
        body="A POST that hit a read timeout was retried automatically and created two orders.",
        comments=[
            "This is by design: retries on non-idempotent methods are opt-in via "
            "`retry_methods`. For safe retries send an `Idempotency-Key` header."
        ],
        resolution="wontfix",
        query="why does a request that errored get submitted again and create duplicates",
    ),
    "http3": dict(
        title="Please support HTTP/3",
        body="Any plans for QUIC / HTTP/3 transport?",
        comments=["Not planned. The underlying socket layer has no stable QUIC support."],
        resolution="wontfix",
        query="can I use QUIC with this client",
    ),
    "corp_cert": dict(
        title="CERTIFICATE_VERIFY_FAILED on corporate network",
        body="Every HTTPS call fails with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer "
        "certificate` inside our office network. Works from home.",
        comments=[
            "Your company's TLS-inspecting proxy re-signs traffic with a private CA. Export that "
            "CA to a PEM file and pass `verify='/path/corp-ca.pem'` or set `ACME_CA_BUNDLE`."
        ],
        resolution="answered",
        query="ssl certificate error only when I am behind my employer's firewall",
    ),
    "json_memory": dict(
        title="Memory grows without bound when decoding large JSON bodies",
        body="Parsing a 800 MB response with `.json()` peaks at 6 GB RSS because the body is "
        "buffered, decoded to str, then parsed.",
        comments=["We can parse incrementally from the byte stream instead of buffering."],
        resolution="fixed",
        file="acme_http/streaming.py",
        fix="Add incremental `iter_json()` and stop double-buffering in `.json()`",
        query="process memory balloons when I parse a huge response",
    ),
    "cookie_persist": dict(
        title="Cookie jar is not saved between sessions",
        body="Cookies set by login are lost on restart; I expected `Session(cookie_file=...)`.",
        comments=["Workaround: pickle `session.cookies` yourself on exit and reload it at start."],
        resolution="open",
        query="stay logged in after restarting my script",
    ),
    "filename_unicode": dict(
        title="Content-Disposition filename*=UTF-8 decoded as latin-1",
        body="Downloads named `résumé.pdf` are saved as `rÃ©sumÃ©.pdf`.",
        comments=["RFC 5987 `filename*` must be percent-decoded as UTF-8."],
        resolution="fixed",
        file="acme_http/streaming.py",
        fix="Decode RFC 5987 filename* as UTF-8",
        query="saved file has garbled accented characters in its name",
    ),
    "async_cancel": dict(
        title="Sockets leak when an async request is cancelled",
        body="Cancelling `await client.get(...)` with `asyncio.wait_for` leaves the connection "
        "in CLOSE_WAIT; after a few hundred cancels we hit the fd limit.",
        comments=["The connection was returned to the pool in a half-read state."],
        resolution="fixed",
        file="acme_http/pool.py",
        fix="Close the connection when a request is cancelled mid-flight",
        query="running out of file descriptors after cancelling many async calls",
    ),
    "dns_cache": dict(
        title="Stale DNS after a failover",
        body="After our load balancer failed over, the client kept talking to the dead IP for hours.",
        comments=["Workaround: call `client.pool.close_idle()` on connection errors."],
        resolution="open",
        query="client keeps connecting to the old server address after we switched servers",
    ),
    "gzip_trailing": dict(
        title="Gzip response with trailing garbage raises BadGzipFile",
        body="A legacy server appends padding after the gzip stream and decoding aborts.",
        comments=["Ignore data after the first complete member, like browsers do."],
        resolution="fixed",
        file="acme_http/streaming.py",
        fix="Tolerate trailing bytes after the first gzip member",
        query="compressed response from an old server fails to decode with a bad file error",
    ),
    "basic_auth": dict(
        title="Basic auth breaks when the password contains a colon or non-ASCII",
        body="`auth=('bob', 'p:ss wörd')` yields 401, curl works.",
        comments=["We encode as latin-1 and split on the first colon incorrectly."],
        resolution="fixed",
        file="acme_http/auth.py",
        fix="Encode basic-auth credentials as UTF-8 and keep colons in the password",
        query="login fails with a password that has special characters",
    ),
    "retry_after": dict(
        title="429 Retry-After header is ignored by the retry policy",
        body="We hammer the API again after 100ms even though it says `Retry-After: 30`.",
        comments=["Backoff should take the max of the computed delay and the header."],
        resolution="fixed",
        file="acme_http/retry.py",
        fix="Honor Retry-After (seconds and HTTP-date) in the retry policy",
        query="the client retries too fast when the server tells it to wait",
    ),
    "win_timeout": dict(
        title="Socket timeout is 1000x too long on Windows",
        body="`timeout=2` waits ~2000 seconds on Windows 11 only.",
        comments=["We pass milliseconds to a socket option that expects seconds there."],
        resolution="fixed",
        file="acme_http/client.py",
        fix="Use seconds for the Windows socket timeout option",
        query="wait time is absurdly long but only on my windows laptop",
    ),
    "py313_warning": dict(
        title="DeprecationWarning on Python 3.13",
        body="`ssl.wrap_socket` usage triggers a DeprecationWarning at import time.",
        comments=["Switch to `SSLContext.wrap_socket`."],
        resolution="fixed",
        file="acme_http/client.py",
        fix="Replace ssl.wrap_socket with SSLContext.wrap_socket",
        query="warning about a deprecated function when importing on the newest python",
    ),
    "ipv6": dict(
        title="IPv6 literal URLs like http://[::1]:8080/ are rejected",
        body="`ValueError: invalid port` for bracketed IPv6 addresses.",
        comments=[],
        resolution="open",
        query="cannot call a server by its ipv6 address",
    ),
    "big_upload": dict(
        title="Multipart upload corrupts files larger than 2 GiB",
        body="The uploaded file's SHA256 differs from the source for 3 GB files.",
        comments=["Content-Length is computed with a signed 32-bit integer."],
        resolution="fixed",
        file="acme_http/streaming.py",
        fix="Use 64-bit lengths for multipart bodies",
        query="large file upload arrives damaged on the server",
    ),
    "metrics": dict(
        title="Expose connection metrics hooks",
        body="It would help to get callbacks for connection acquire/release times.",
        comments=[],
        resolution="open",
        query="I want to measure how long requests wait for a connection",
    ),
    "no_proxy_cidr": dict(
        title="NO_PROXY ignores CIDR ranges like 10.0.0.0/8",
        body="Internal addresses still go through the corporate proxy although NO_PROXY lists the range.",
        comments=["We only compare hostnames and exact IPs today."],
        resolution="fixed",
        file="acme_http/auth.py",
        fix="Match NO_PROXY entries that are CIDR ranges",
        query="internal servers still go through the proxy even though I excluded their network",
    ),
}

# (duplicate_key, canonical_key, title, body)
DUPLICATES: list[tuple[str, str, str, str]] = [
    (
        "dup_pool",
        "pool",
        "Client freezes with many threads",
        "Heavy multithreaded use eventually blocks all calls.",
    ),
    (
        "dup_stream_timeout",
        "stream_timeout",
        "Timeout not applied to large files",
        "Setting timeout has no effect on a 2 GB download.",
    ),
    (
        "dup_cert",
        "corp_cert",
        "SSL: unable to get local issuer certificate",
        "Fails at work, fine on mobile hotspot.",
    ),
    (
        "dup_memory",
        "json_memory",
        "OOM killed when reading big API response",
        "Container with 4 GB gets OOM-killed on a 600 MB payload.",
    ),
    (
        "dup_cancel",
        "async_cancel",
        "Too many open files after asyncio timeouts",
        "Running wait_for in a loop exhausts file descriptors.",
    ),
    (
        "dup_retry_after",
        "retry_after",
        "Rate limiting: client ignores server's wait hint",
        "429s keep coming because backoff is too short.",
    ),
    (
        "dup_basic",
        "basic_auth",
        "401 with passwords containing special characters",
        "Same credentials work in curl.",
    ),
    (
        "dup_win",
        "win_timeout",
        "Windows: timeouts are in the wrong unit",
        "Timeouts take thousands of seconds.",
    ),
]

# Q&A discussions with an accepted answer: key -> (title, body, answer, query)
DISCUSSIONS: dict[str, dict] = {
    "disc_retries": dict(
        title="How do I configure retries with exponential backoff?",
        body="I want 3 retries, doubling the delay each time, only for 502/503.",
        answer="Use `RetryPolicy(total=3, backoff_factor=0.5, status_forcelist={502, 503})` and "
        "pass it as `Client(retry=policy)`.",
        query="how can I make failed calls try again with increasing pauses",
    ),
    "disc_threads": dict(
        title="Is one Client safe to share across threads?",
        body="Should every thread create its own client or can I share one?",
        answer="Share one `Client`: it is thread-safe and pools connections. Create one per "
        "process, not per request.",
        query="can several threads use the same client object",
    ),
    "disc_generator_upload": dict(
        title="Can I upload from a generator without loading everything into memory?",
        body="I produce rows lazily and want to stream them as the request body.",
        answer="Pass the generator as `data=`; it is sent with chunked transfer encoding.",
        query="send a request body that is produced piece by piece",
    ),
    "disc_mock": dict(
        title="Best way to mock the client in unit tests?",
        body="I do not want network calls in CI.",
        answer="Use `acme_http.testing.MockTransport` and pass `transport=` to `Client`.",
        query="how do I avoid real network access in my tests",
    ),
    "disc_total_timeout": dict(
        title="Why does my timeout not limit the whole request?",
        body="timeout=5 but a slow response took 40 seconds in total.",
        answer="`timeout` applies per socket operation. Use `Timeout(total=5)` for an overall deadline.",
        query="a hard cap on the entire duration of a request",
    ),
    "disc_socks": dict(
        title="Using the client through a SOCKS proxy",
        body="Is SOCKS5 supported?",
        answer="Yes: `pip install acme-http[socks]` and set `proxy='socks5://host:1080'`.",
        query="route traffic through a socks5 server",
    ),
}

INITIAL_FILES: dict[str, str] = {
    "acme_http/__init__.py": '"""acme_http: a fictional HTTP client used only as a test fixture."""\n',
    "acme_http/client.py": "# known-issue: win_timeout\n# known-issue: py313_warning\nclass Client: ...\n",
    "acme_http/pool.py": "# known-issue: pool\n# known-issue: async_cancel\nPOOL_MAXSIZE = 10\n",
    "acme_http/auth.py": "# known-issue: proxy_redirect\n# known-issue: basic_auth\n# known-issue: no_proxy_cidr\n",
    "acme_http/retry.py": "# known-issue: retry_after\nclass RetryPolicy: ...\n",
    "acme_http/streaming.py": (
        "# known-issue: stream_timeout\n# known-issue: json_memory\n# known-issue: filename_unicode\n"
        "# known-issue: gzip_trailing\n# known-issue: big_upload\n"
    ),
}

README = """# langchain-github-testbed

A **synthetic fixture** for evaluating [langchain-github](https://github.com/daivikawasthi01/langchain-github).

`acme_http` is a fictional HTTP client. Every issue, pull request and discussion here was generated
by `eval/testbed/seed.py` so that retrieval quality can be measured against known answers
(duplicates, issues closed by a PR, wontfix decisions, answered discussions). Nothing here is a
real library or real bug report. Please don't open issues.
"""
