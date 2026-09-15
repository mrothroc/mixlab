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

Usage:
  DOCKERHUB_TOKEN=... python3 docker/sync_dockerhub_description.py \
      --user michaelrothrock --repo mixlab --file docker/DOCKERHUB.md
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

LOGIN_URL = "https://hub.docker.com/v2/users/login/"
REPO_URL = "https://hub.docker.com/v2/repositories/{user}/{repo}/"

# Docker Hub silently truncates past this rather than returning an error, which
# would publish a page ending mid-sentence with no signal that it happened.
MAX_DESCRIPTION_CHARS = 25000

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


def build_payload(description: str) -> dict:
    """The PATCH body. Only full_description — never touch other repo fields."""
    return {"full_description": description}


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


def publish(user: str, repo: str, description: str, jwt: str) -> None:
    url = REPO_URL.format(user=user, repo=repo)
    try:
        status = _patch_json(url, build_payload(description),
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
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the file and exit without contacting Docker Hub")
    args = parser.parse_args()

    token = os.environ.get("DOCKERHUB_TOKEN", "")

    try:
        # Validate BEFORE the credential check so a broken file fails the build
        # even on a run that would not have published anything.
        description = load_description(args.file)
    except DescriptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"dry-run OK: {args.file} is {len(description)} chars "
              f"(limit {MAX_DESCRIPTION_CHARS})")
        return 0

    if should_skip(args.user, token):
        print("Skipping Docker Hub description sync "
              "(DOCKERHUB_USER or DOCKERHUB_TOKEN not set)")
        return 0

    try:
        publish(args.user, args.repo, description, login(args.user, token))
    except DescriptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"ERROR: could not reach Docker Hub: {exc.reason}", file=sys.stderr)
        return 1

    print(f"Updated {args.user}/{args.repo} overview "
          f"({len(description)} chars) from {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
