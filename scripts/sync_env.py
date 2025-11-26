import os
from pathlib import Path
from typing import Dict

EXAMPLE_ENV_PATH = Path(os.environ.get("EXAMPLE_ENV_PATH", "/app/env.example"))
TARGET_ENV_PATH = Path(os.environ.get("TARGET_ENV_PATH", "/app/.env"))


def parse_env_file(path: Path) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    if not path.exists():
        return variables

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        variables[key.strip()] = value
    return variables


def append_missing_variables(example_vars: Dict[str, str], target_vars: Dict[str, str]) -> int:
    missing = {key: value for key, value in example_vars.items() if key not in target_vars}
    if not missing:
        return 0

    TARGET_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_ENV_PATH.open("a", encoding="utf-8") as env_file:
        if TARGET_ENV_PATH.stat().st_size > 0:
            env_file.write("\n")
        for key, value in missing.items():
            env_file.write(f"{key}={value}\n")
    return len(missing)


def main() -> None:
    if not EXAMPLE_ENV_PATH.exists():
        raise FileNotFoundError(f"Example env file not found at {EXAMPLE_ENV_PATH}")

    example_vars = parse_env_file(EXAMPLE_ENV_PATH)
    target_vars = parse_env_file(TARGET_ENV_PATH)

    added = append_missing_variables(example_vars, target_vars)
    message = "No new variables found. .env is up to date." if added == 0 else f"Added {added} missing variable(s) to {TARGET_ENV_PATH}."
    print(message)


if __name__ == "__main__":
    main()
