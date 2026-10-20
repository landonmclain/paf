"""One list of credential-bearing files, enforced on both sides of the boundary.

Two mechanisms keep these out of the model's reach, and they must agree:

- the Docker sandbox blanks them (``DockerExecutor`` ``exclude``: the real file
  stays on the host, an empty one is mounted over it inside the container);
- the host-side file tools refuse to read them (``FileSystem.denied_patterns``),
  because the sandbox overlay does not cover tools that run on the host.

Names are matched at ANY depth (``backend/.env`` is as secret as ``.env``).
Deliberately conservative about source code: ``secrets.yaml`` is listed but a
glob like ``secrets*`` is not, since that would hide ``secrets_manager.py``.

``.env.example`` / ``.env.sample`` are blanked/denied along with the rest
(``.env.*``); an agent that needs the template must be given it another way.
"""

from __future__ import annotations

# Basename globs for files.
SECRET_FILE_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".pgpass",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "credentials.json",
    "secrets.yaml",
    "secrets.yml",
    "secrets.json",
    "secrets.toml",
)

# Directory names whose whole contents are off limits.
SECRET_DIRS: tuple[str, ...] = (".git", ".ssh", ".aws", ".gnupg")


def sandbox_excludes() -> tuple[str, ...]:
    """``DockerExecutor`` ``exclude`` patterns (basename globs, any depth)."""
    return (*SECRET_FILE_GLOBS, *SECRET_DIRS)


def fs_denied_patterns() -> list[str]:
    """``FileSystem.denied_patterns`` for the same set.

    The file tools match a pattern against the whole relative path with
    ``fnmatch`` (whose ``*`` crosses ``/``), so each name is given bare (root
    level) and with a ``**/`` prefix (any depth); directories also get ``/*`` for
    their contents.
    """
    patterns: list[str] = []
    for name in SECRET_FILE_GLOBS:
        patterns += [name, f"**/{name}"]
    for name in SECRET_DIRS:
        patterns += [name, f"{name}/*", f"**/{name}", f"**/{name}/*"]
    return patterns
