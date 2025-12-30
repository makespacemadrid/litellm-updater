"""Routing group management API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import get_session
from shared.crud import get_config
from backend.litellm_client import push_routing_groups_to_litellm, list_routing_group_deployments
from shared.crud import (
    get_routing_groups,
    get_routing_group,
    create_routing_group,
    update_routing_group,
    delete_routing_group,
    replace_routing_targets,
    list_routing_candidates,
    compile_routing_groups,
)
from shared.db_models import RoutingGroup
from shared.tags import normalize_tags

router = APIRouter()


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [item.strip() for item in raw.split(",")]
    return [p for p in parts if p]


def _normalize_key(value: str | None) -> str:
    if not value:
        return ""
    normalized = normalize_tags([value])
    return normalized[0] if normalized else value.strip().lower()


def _target_key(provider_name: str | None, model_id: str | None) -> str:
    return f"{_normalize_key(provider_name)}::{_normalize_key(model_id)}"


class RoutingTargetPayload(BaseModel):
    provider_id: int
    model_id: str
    weight: int = 1
    priority: int = 0
    enabled: bool = True


class RoutingGroupPayload(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    targets: list[RoutingTargetPayload] = Field(default_factory=list)


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
    await session.flush()
    group = await get_routing_group(session, group.id)
    if not group:
        raise HTTPException(404, "Routing group not found after create")
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


@router.post("/push")
async def push_all_groups(session: AsyncSession = Depends(get_session)) -> dict:
    """Push all routing groups to LiteLLM."""
    config = await get_config(session)
    if not config.litellm_base_url:
        raise HTTPException(400, "LiteLLM destination not configured")
    stats = await push_routing_groups_to_litellm(session, config)
    return {"status": "ok", "stats": stats}


@router.get("/status")
async def routing_group_status(session: AsyncSession = Depends(get_session)) -> dict:
    """Return routing group status compared to LiteLLM."""
    config = await get_config(session)
    if not config.litellm_base_url:
        raise HTTPException(400, "LiteLLM destination not configured")

    groups = await get_routing_groups(session)
    groups = [await get_routing_group(session, group.id) for group in groups]
    groups = [group for group in groups if group is not None]

    litellm_entries = await list_routing_group_deployments(config)
    litellm_by_group: dict[str, list[dict]] = {}
    for entry in litellm_entries:
        litellm_by_group.setdefault(entry["group"], []).append(entry)

    response_groups: list[dict] = []
    db_group_names = {group.name for group in groups}

    for group in groups:
        db_targets = []
        db_keys = set()
        for target in sorted(group.targets, key=lambda t: (t.priority, t.id)):
            provider_name = target.provider.name if target.provider else None
            db_targets.append(
                {
                    "provider_name": provider_name,
                    "model_id": target.model_id,
                    "enabled": target.enabled,
                }
            )
            if target.enabled:
                db_keys.add(_target_key(provider_name, target.model_id))

        litellm_targets = litellm_by_group.get(group.name, [])
        litellm_keys = {
            _target_key(entry.get("provider"), entry.get("model_id"))
            for entry in litellm_targets
            if entry.get("provider") and entry.get("model_id")
        }

        missing_in_litellm = [
            target for target in db_targets
            if target["enabled"] and _target_key(target["provider_name"], target["model_id"]) not in litellm_keys
        ]
        extra_in_litellm = [
            entry for entry in litellm_targets
            if _target_key(entry.get("provider"), entry.get("model_id")) not in db_keys
        ]

        response_groups.append(
            {
                "id": group.id,
                "name": group.name,
                "description": group.description,
                "db_targets": db_targets,
                "db_count": len([t for t in db_targets if t["enabled"]]),
                "litellm_count": len(litellm_targets),
                "litellm_targets": litellm_targets,
                "missing_in_litellm": missing_in_litellm,
                "extra_in_litellm": extra_in_litellm,
            }
        )

    litellm_only = [
        {
            "name": name,
            "litellm_count": len(entries),
            "litellm_targets": entries,
        }
        for name, entries in litellm_by_group.items()
        if name not in db_group_names
    ]

    return {"groups": response_groups, "litellm_only": litellm_only}


@router.get("/{group_id}")
async def get_group(group_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """Get a single routing group."""
    group = await get_routing_group(session, group_id)
    if not group:
        raise HTTPException(404, "Routing group not found")
    group = await get_routing_group(session, group.id)
    if not group:
        raise HTTPException(404, "Routing group not found after update")
    return _group_to_dict(group)


@router.post("/{group_id}/push")
async def push_group(group_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """Push a single routing group to LiteLLM."""
    config = await get_config(session)
    if not config.litellm_base_url:
        raise HTTPException(400, "LiteLLM destination not configured")
    stats = await push_routing_groups_to_litellm(session, config, group_id=group_id)
    return {"status": "ok", "stats": stats}


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
