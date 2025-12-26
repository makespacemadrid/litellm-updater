#!/usr/bin/env python3
"""Quick script to check available models in the database."""
import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from shared.db_models import Model, Provider


async def check_models():
    """Check models in the database."""
    engine = create_async_engine("sqlite+aiosqlite:///data/models.db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        result = await session.execute(
            select(Model, Provider)
            .join(Provider)
            .where(
                Provider.type.notin_(["compat"]),
                Model.is_orphaned == False
            )
            .order_by(Provider.name, Model.model_id)
        )

        rows = result.all()

        print(f"\n{'Provider':<20} {'Model ID':<40} {'Context':<10} {'Vision':<10} {'Reasoning':<10} {'Type':<15}")
        print("=" * 115)

        for model, provider in rows:
            params = model.litellm_params_dict or {}
            context = model.context_window or params.get("max_tokens", "?")
            vision = "Yes" if params.get("supports_vision") else "-"
            reasoning = "Yes" if params.get("supports_reasoning") else "-"
            model_type = model.model_type or "-"

            print(f"{provider.name:<20} {model.model_id:<40} {str(context):<10} {vision:<10} {reasoning:<10} {model_type:<15}")

        print(f"\nTotal: {len(rows)} models")


if __name__ == "__main__":
    asyncio.run(check_models())
