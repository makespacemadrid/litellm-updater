"""Routing group management API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import get_session
from shared.crud import (
    get_routing_groups,
    get_routing_group,
    create_routing_group,
    update_routing_group,
    delete_routing_group,
    replace_routing_targets,
    replace_provider_limits,
    list_routing_candidates,
    compile_routing_groups,
)
from shared.db_models import RoutingGroup

router = APIRouter()


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [item.strip() for item in raw.split(",")]
    return [p for p in parts if p]


class RoutingTargetPayload(BaseModel):
    provider_id: int
    model_id: str
    weight: int = 1
    priority: int = 0
    enabled: bool = True


class ProviderLimitPayload(BaseModel):
    provider_id: int
    max_requests_per_hour: int | None = None


class RoutingGroupPayload(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    targets: list[RoutingTargetPayload] = Field(default_factory=list)
    provider_limits: list[ProviderLimitPayload] = Field(default_factory=list)


def _group_to_dict(group: RoutingGroup) -> dict:
    return {
        "id": group.id,
        "name": group.name,
        "description": group.description,
        "capabilities": group.capabilities_list,
        "targets": [
            {
                "id": target.id,
                "provider_id": target.provider_id,
                "provider_name": target.provider.name if target.provider else None,
                "provider_type": target.provider.type if target.provider else None,
                "model_id": target.model_id,
                "weight": target.weight,
                "priority": target.priority,
                "enabled": target.enabled,
            }
            for target in sorted(group.targets, key=lambda t: (t.priority, t.id))
        ],
        "provider_limits": [
            {
                "id": limit.id,
                "provider_id": limit.provider_id,
                "provider_name": limit.provider.name if limit.provider else None,
                "max_requests_per_hour": limit.max_requests_per_hour,
            }
            for limit in sorted(group.provider_limits, key=lambda p: p.provider_id)
        ],
    }


@router.get("")
@router.get("/")
async def list_groups(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all routing groups."""
    groups = await get_routing_groups(session)
    return [
        {
            "id": group.id,
            "name": group.name,
            "description": group.description,
            "capabilities": group.capabilities_list,
        }
        for group in groups
    ]


@router.post("")
@router.post("/")
async def create_group(
    payload: RoutingGroupPayload,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Create a routing group with targets and provider limits."""
    existing = await session.execute(
        select(RoutingGroup).where(RoutingGroup.name == payload.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, "Routing group name already exists")

    group = await create_routing_group(
        session,
        name=payload.name,
        description=payload.description,
        capabilities=payload.capabilities,
    )
    await replace_routing_targets(
        session, group, [target.model_dump() for target in payload.targets]
    )
    await replace_provider_limits(
        session, group, [limit.model_dump() for limit in payload.provider_limits]
    )
    await session.flush()
    return _group_to_dict(group)


@router.get("/candidates")
async def list_candidates(
    capabilities: str | None = Query(default=None),
    query: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List candidate models filtered by capabilities."""
    caps = _parse_csv(capabilities)
    return await list_routing_candidates(session, capabilities=caps, query=query)


@router.get("/compiled")
async def compiled_config(session: AsyncSession = Depends(get_session)) -> dict:
    """Return compiled routing group configuration."""
    return {"groups": await compile_routing_groups(session)}


@router.get("/{group_id}")
async def get_group(group_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """Get a single routing group."""
    group = await get_routing_group(session, group_id)
    if not group:
        raise HTTPException(404, "Routing group not found")
    return _group_to_dict(group)


@router.put("/{group_id}")
async def update_group(
    group_id: int,
    payload: RoutingGroupPayload,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update a routing group and replace targets/limits."""
    group = await get_routing_group(session, group_id)
    if not group:
        raise HTTPException(404, "Routing group not found")

    existing = await session.execute(
        select(RoutingGroup).where(RoutingGroup.name == payload.name, RoutingGroup.id != group_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, "Routing group name already exists")

    try:
        await update_routing_group(
            session,
            group,
            name=payload.name,
            description=payload.description,
            capabilities=payload.capabilities,
        )
        await replace_routing_targets(
            session, group, [target.model_dump() for target in payload.targets]
        )
        await replace_provider_limits(
            session, group, [limit.model_dump() for limit in payload.provider_limits]
        )
        await session.flush()
    except IntegrityError as exc:
        raise HTTPException(400, "Invalid routing group payload") from exc

    return _group_to_dict(group)


@router.delete("/{group_id}")
async def remove_group(group_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete a routing group."""
    group = await get_routing_group(session, group_id, include_children=False)
    if not group:
        raise HTTPException(404, "Routing group not found")
    await delete_routing_group(session, group)
    return {"status": "ok"}
