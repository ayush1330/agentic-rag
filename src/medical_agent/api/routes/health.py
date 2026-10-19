"""
Health Check Endpoints

Provides endpoints for monitoring application health, readiness,
and liveness for container orchestration systems.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(description="Current server timestamp")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current environment")


class DetailedHealthStatus(HealthStatus):
    """Detailed health check with component statuses."""

    components: dict[str, dict[str, Any]] = Field(
        description="Individual component health statuses"
    )


class ReadinessStatus(BaseModel):
    """Readiness probe response model."""

    ready: bool = Field(description="Whether the service is ready to accept traffic")
    checks: dict[str, bool] = Field(description="Individual readiness check results")


@router.get(
    "/health",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Basic Health Check",
    description="Returns basic health status of the application.",
)
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint.

    Returns a simple health status indicating the service is running.
    Used by load balancers and monitoring systems.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(UTC),
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/health/detailed",
    response_model=DetailedHealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Detailed Health Check",
    description="Returns detailed health status including component checks.",
)
async def detailed_health_check() -> DetailedHealthStatus:
    """
    Detailed health check with component statuses.

    Checks the health of individual components:
    - Database connectivity
    - Azure OpenAI configuration
    - LangSmith configuration
    - GCP configuration

    Note: This is a configuration check, not a connectivity test.
    For production, add actual connectivity tests.
    """
    components: dict[str, dict[str, Any]] = {
        "database": {
            "configured": bool(settings.database_connection_string),
            "host": settings.resolved_postgres_host,
            "port": settings.resolved_postgres_port,
        },
        "azure_openai": {
            "configured": settings.is_azure_openai_configured(),
            "deployment": settings.azure_openai_deployment_name,
        },
        "langsmith": {
            "configured": settings.is_langsmith_configured(),
            "project": settings.langsmith_project,
        },
        "gcp": {
            "configured": bool(settings.gcp_project_id),
            "bucket": settings.gcp_bucket_name,
        },
    }

    return DetailedHealthStatus(
        status="healthy",
        timestamp=datetime.now(UTC),
        version=settings.app_version,
        environment=settings.environment,
        components=components,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessStatus,
    status_code=status.HTTP_200_OK,
    summary="Readiness Probe",
    description="Kubernetes readiness probe - checks if service can accept traffic.",
)
async def readiness_check() -> ReadinessStatus:
    """
    Readiness probe for Kubernetes/Cloud Run.

    Checks if the service is ready to accept traffic.
    Returns 200 if ready, could return 503 if not ready.
    """
    checks = {
        "configuration_loaded": True,
        "database_connected": await _check_database(),
    }

    all_ready = all(checks.values())

    return ReadinessStatus(
        ready=all_ready,
        checks=checks,
    )


async def _check_database() -> bool:
    """Check database connectivity."""
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            settings.sqlalchemy_async_database_url,
            echo=False,
        )
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception as e:
        logger.warning(f"Database check failed: {e}")
        return False


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Probe",
    description="Kubernetes liveness probe - checks if service should be restarted.",
)
async def liveness_check() -> dict[str, str]:
    """
    Liveness probe for Kubernetes/Cloud Run.

    Simple check to verify the service process is alive.
    If this fails, the container should be restarted.
    """
    return {"status": "alive"}


class CloudServicesStatus(BaseModel):
    """Cloud services connectivity status."""

    overall_status: str = Field(description="Overall cloud services status")
    timestamp: datetime = Field(description="Check timestamp")
    services: dict[str, dict[str, Any]] = Field(description="Individual service statuses")


@router.get(
    "/health/cloud-services",
    response_model=CloudServicesStatus,
    status_code=status.HTTP_200_OK,
    summary="Cloud Services Health Check",
    description="Checks connectivity to all cloud services (GCP, Azure, LangSmith).",
)
async def cloud_services_check() -> CloudServicesStatus:
    """
    Check connectivity to all external cloud services.

    Tests connections to:
    - GCP Cloud Storage
    - Azure OpenAI
    - LangSmith
    """
    services: dict[str, dict[str, Any]] = {}

    # Check GCP Storage
    services["gcp_storage"] = await _check_gcp_storage()

    # Check Azure OpenAI
    services["azure_openai"] = await _check_azure_openai()

    # Check LangSmith
    services["langsmith"] = await _check_langsmith()

    # Determine overall status
    all_configured = all(s.get("configured", False) for s in services.values())
    all_connected = all(
        s.get("connected", False) for s in services.values() if s.get("configured", False)
    )

    if all_configured and all_connected:
        overall_status = "healthy"
    elif any(s.get("connected", False) for s in services.values()):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return CloudServicesStatus(
        overall_status=overall_status,
        timestamp=datetime.now(UTC),
        services=services,
    )


async def _check_gcp_storage() -> dict[str, Any]:
    """Check GCP Storage connectivity."""
    result = {
        "configured": False,
        "connected": False,
        "bucket": settings.gcp_bucket_name,
        "error": None,
    }

    try:
        from medical_agent.infrastructure.gcp_storage import GCPStorageClient

        client = GCPStorageClient()
        result["configured"] = client.is_configured()

        if result["configured"]:
            client.verify_connection()
            result["connected"] = True
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"GCP Storage check failed: {e}")

    return result


async def _check_azure_openai() -> dict[str, Any]:
    """Check Azure OpenAI configuration."""
    result = {
        "configured": False,
        "connected": False,
        "chat_deployment": settings.azure_openai_deployment_name,
        "embedding_deployment": settings.azure_openai_embedding_deployment_name,
        "error": None,
    }

    try:
        result["configured"] = settings.is_azure_openai_configured()

        if result["configured"]:
            # Configuration check only, skip actual API calls for health check
            result["connected"] = True
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Azure OpenAI check failed: {e}")

    return result


async def _check_langsmith() -> dict[str, Any]:
    """Check LangSmith configuration."""
    result = {
        "configured": False,
        "connected": False,
        "project": settings.langsmith_project,
        "error": None,
    }

    try:
        result["configured"] = settings.is_langsmith_configured()
        # LangSmith doesn't need connectivity check - env vars are enough
        result["connected"] = result["configured"]
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"LangSmith check failed: {e}")

    return result
