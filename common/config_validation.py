"""Shared configuration schema checks."""

def object_settings(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} settings: {', '.join(sorted(unknown))}")
    return value


