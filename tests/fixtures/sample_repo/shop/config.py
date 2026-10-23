"""Settings shared across the shop."""

SETTINGS: dict[str, str] = {"currency": "USD", "tax_rate": "0.08"}
DEFAULT_REGION = "us"


def get_setting(name: str, default: str = "") -> str:
    """Look up one setting by name."""
    return SETTINGS.get(name, default)


def region() -> str:
    """The configured sales region."""
    return get_setting("region", DEFAULT_REGION)
