#!/usr/bin/env python3
"""Quick script to check available models in the database using sqlite3."""
import sqlite3
import json

def check_models():
    """Check models in the database."""
    conn = sqlite3.connect("data/models.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            p.name as provider,
            m.model_id,
            m.context_window,
            m.litellm_params,
            m.model_type
        FROM models m
        JOIN providers p ON m.provider_id = p.id
        WHERE p.type != 'compat'
          AND m.is_orphaned = 0
        ORDER BY p.name, m.model_id
    """)

    rows = cursor.fetchall()

    print(f"\n{'Provider':<20} {'Model ID':<40} {'Context':<10} {'Vision':<10} {'Reasoning':<10} {'Type':<15}")
    print("=" * 115)

    for provider, model_id, context, litellm_params_json, model_type in rows:
        try:
            params = json.loads(litellm_params_json) if litellm_params_json else {}
        except:
            params = {}

        context_str = str(context) if context else str(params.get("max_tokens", "?"))
        vision = "Yes" if params.get("supports_vision") else "-"
        reasoning = "Yes" if params.get("supports_reasoning") else "-"
        model_type_str = model_type or "-"

        print(f"{provider:<20} {model_id:<40} {context_str:<10} {vision:<10} {reasoning:<10} {model_type_str:<15}")

    print(f"\nTotal: {len(rows)} models")

    conn.close()


if __name__ == "__main__":
    check_models()
