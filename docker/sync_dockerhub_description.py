#!/usr/bin/env python3
"""Publish docker/DOCKERHUB.md as the Docker Hub repository overview.

WHY THIS EXISTS: the overview is the landing page for every `docker pull`, and
Docker Hub only lets you edit it in a web form. A hand-edited form field drifts
from the repository the moment either one changes, and nothing catches it. Making
the repo the source of truth and pushing it from the same build that pushes the
image means the page cannot describe an older version of the software than the
one it is sitting next to.

Runs in the CI build that already authenticates to Docker Hub, so it needs no
credential of its own beyond the token that step uses.

FAILS LOUDLY on a degenerate description. Docker Hub accepts an empty
full_description without complaint, so a missing or blank file would blank a
working overview and exit 0. Every such case raises here instead.

The access token is read from an environment variable rather than an argument so
it never appears in a process listing or a build log. --token-env names that
variable: Cloud Build refuses to bind one secret version to two env names, so
the step reuses the DOCKER_TOKEN binding the image push already declares.

Usage:
  DOCKERHUB_TOKEN=... python3 docker/sync_dockerhub_description.py \
      --user michaelrothrock --repo mixlab --file docker/DOCKERHUB.md
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

LOGIN_URL = "https://hub.docker.com/v2/users/login/"
REPO_URL = "https://hub.docker.com/v2/repositories/{user}/{repo}/"

# Docker Hub silently truncates past this rather than returning an error, which
# would publish a page ending mid-sentence with no signal that it happened.
MAX_DESCRIPTION_CHARS = 25000

# Docker Hub's one-line description, shown in search results and on the
# repository card. Longer values are rejected by the API.
MAX_SHORT_DESCRIPTION_CHARS = 100

# The one-liner lives in the same file as the overview so the two are edited
# together. An HTML comment renders as nothing, and it is stripped before the
# body is published.
SHORT_MARKER = re.compile(r"^[ \t]*<!--[ \t]*short:(.*?)-->[ \t]*\r?\n?", re.M | re.S)

TIMEOUT_SECONDS = 30


class DescriptionError(Exception):
    """The description is unusable; abort rather than publish it."""


def load_description(path: str) -> str:
    """Read the overview, refusing anything that would blank the page."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError as exc:
        raise DescriptionError(f"description file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise DescriptionError(f"description is not valid UTF-8: {path}") from exc

    if not text.strip():
        raise DescriptionError(f"description is empty: {path}")
    if len(text) > MAX_DESCRIPTION_CHARS:
        raise DescriptionError(
            f"description is {len(text)} chars, over Docker Hub's "
            f"{MAX_DESCRIPTION_CHARS} limit: {path}"
        )
    return text


def split_description(text: str) -> tuple:
    """Split an overview file into (short description, body).

    The short description is REQUIRED. Docker Hub keeps it in a separate field
    that is easy to forget, and a forgotten one stays published and stale — which
    is exactly what happened to mixlab-cuda's sm_80/86/89 line after the overview
    was corrected.
    """
    match = SHORT_MARKER.search(text)
    if not match:
        raise DescriptionError(
            "no short description found: add a line like "
            "'<!-- short: One line shown in Docker Hub search results. -->'")
    short = match.group(1).strip()
    if not short:
        raise DescriptionError("short description marker is empty")
    if len(short) > MAX_SHORT_DESCRIPTION_CHARS:
        raise DescriptionError(
            f"short description is {len(short)} chars, over Docker Hub's "
            f"{MAX_SHORT_DESCRIPTION_CHARS} limit: {short[:60]}...")

    body = SHORT_MARKER.sub("", text, count=1)
    if not body.strip():
        raise DescriptionError("nothing left after removing the short-description marker")
    return short, body


def build_payload(description: str, short: str = None) -> dict:
    """The PATCH body. Only the two description fields — nothing else is touched."""
    payload = {"full_description": description}
    if short is not None:
        payload["description"] = short
    return payload


def parse_target(spec: str) -> tuple:
    """Parse a "repo=path" target.

    One flag per repository keeps the published set visible as a single list in
    the build config, rather than three near-identical steps that can drift.
    """
    repo, separator, path = spec.partition("=")
    if not separator or not repo.strip() or not path.strip():
        raise DescriptionError(
            f'--target must look like "repo=path/to/file.md", got: {spec!r}')
    return repo.strip(), path.strip()


def read_token(env_name: str) -> str:
    """Read the access token from the named environment variable.

    Kept out of argv deliberately: an argument would be visible in `ps` output
    and in Cloud Build's step logs.
    """
    if not env_name:
        raise DescriptionError("--token-env must name an environment variable")
    return os.environ.get(env_name, "")


def should_skip(user: str, token: str) -> bool:
    """Mirror the image-push step: no credentials configured means no-op."""
    return not user or not token


def _post_json(url: str, body: dict, headers: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _patch_json(url: str, body: dict, headers: dict) -> int:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="PATCH")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.status


def login(user: str, token: str) -> str:
    """Exchange a Docker Hub personal access token for a JWT."""
    try:
        payload = _post_json(LOGIN_URL, {"username": user, "password": token}, {})
    except urllib.error.HTTPError as exc:
        # Never echo the token. The status alone identifies the problem.
        raise DescriptionError(
            f"Docker Hub login failed with HTTP {exc.code}. If this is 401 or 403, "
            "the access token may lack repository write scope."
        ) from exc
    jwt = payload.get("token")
    if not jwt:
        raise DescriptionError("Docker Hub login returned no token")
    return jwt


def publish(user: str, repo: str, description: str, jwt: str,
            short: str = None) -> None:
    url = REPO_URL.format(user=user, repo=repo)
    try:
        status = _patch_json(url, build_payload(description, short),
                             {"Authorization": f"JWT {jwt}"})
    except urllib.error.HTTPError as exc:
        raise DescriptionError(
            f"updating {user}/{repo} failed with HTTP {exc.code}"
        ) from exc
    if status != 200:
        raise DescriptionError(f"unexpected status {status} updating {user}/{repo}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish a Markdown file as a Docker Hub repository overview.")
    parser.add_argument("--user", default=os.environ.get("DOCKERHUB_USER", ""))
    parser.add_argument("--repo", default="mixlab")
    parser.add_argument("--file", default="docker/DOCKERHUB.md")
    parser.add_argument("--target", action="append", default=[], metavar="REPO=FILE",
                        help="publish FILE as REPO's overview; repeatable. "
                             "Supersedes --repo/--file when given.")
    parser.add_argument("--token-env", default="DOCKERHUB_TOKEN",
                        help="name of the env var holding the access token "
                             "(default: DOCKERHUB_TOKEN)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the file and exit without contacting Docker Hub")
    args = parser.parse_args()

    token = read_token(args.token_env)

    try:
        targets = ([parse_target(spec) for spec in args.target]
                   if args.target else [(args.repo, args.file)])
        # Validate EVERY file before publishing ANY of them, and before the
        # credential check: one broken overview should fail the build without
        # leaving the other repositories half-updated.
        loaded = []
        for repo, path in targets:
            short, body = split_description(load_description(path))
            loaded.append((repo, path, body, short))
    except DescriptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        for repo, path, body, short in loaded:
            print(f"dry-run OK: {path} -> {repo} is {len(body)} chars "
                  f"(limit {MAX_DESCRIPTION_CHARS})")
            print(f"            short ({len(short)}/{MAX_SHORT_DESCRIPTION_CHARS}): {short}")
        return 0

    if should_skip(args.user, token):
        print(f"Skipping Docker Hub description sync "
              f"(--user or ${args.token_env} not set)")
        return 0

    try:
        jwt = login(args.user, token)
    except (DescriptionError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Attempt every repository even if one fails, so a single bad repo name does
    # not silently leave the rest of the set unpublished.
    failed = False
    for repo, path, body, short in loaded:
        try:
            publish(args.user, repo, body, jwt, short)
        except (DescriptionError, urllib.error.URLError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"Updated {args.user}/{repo} overview "
              f"({len(body)} chars) + short description, from {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
