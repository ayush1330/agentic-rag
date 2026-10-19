#!/usr/bin/env python3
"""
Infrastructure Setup and Verification Script

This script helps set up and verify cloud infrastructure for the
FemTech Medical RAG Agent. It provides:

1. Step-by-step setup instructions
2. Connection verification for all services
3. Health checks for GCP, Azure, and LangSmith

Usage:
    python scripts/setup_infrastructure.py --check-all
    python scripts/setup_infrastructure.py --check gcp
    python scripts/setup_infrastructure.py --check azure
    python scripts/setup_infrastructure.py --check langsmith
    python scripts/setup_infrastructure.py --setup-guide
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} ---")


def print_status(name: str, status: bool, message: str = "") -> None:
    """Print a status line."""
    icon = "✅" if status else "❌"
    msg = f" - {message}" if message else ""
    print(f"  {icon} {name}{msg}")


def check_env_file() -> bool:
    """Check if .env file exists."""
    env_path = project_root / ".env"
    env_example_path = project_root / ".env.example"
    
    if not env_path.exists():
        print_status(".env file", False, "Not found")
        if env_example_path.exists():
            print("     → Copy .env.example to .env and fill in your values")
        else:
            print("     → Create .env file with required environment variables")
        return False
    
    print_status(".env file", True, "Found")
    return True


def check_gcp_configuration() -> dict[str, bool]:
    """Check GCP configuration and connectivity."""
    print_section("GCP Cloud Storage")
    
    results = {"configured": False, "connected": False, "bucket_exists": False}
    
    try:
        from medical_agent.core.config import settings
        from medical_agent.infrastructure.gcp_storage import GCPStorageClient
        
        client = GCPStorageClient()
        
        # Check configuration
        if client.is_configured():
            print_status("Configuration", True, f"Project: {client.project_id}")
            results["configured"] = True
        else:
            print_status("Configuration", False, "GCP_PROJECT_ID not set")
            return results
        
        # Check bucket
        print_status("Bucket name", True, client.bucket_name)
        
        # Try to connect
        try:
            client.verify_connection()
            print_status("Bucket connection", True)
            results["connected"] = True
            results["bucket_exists"] = True
        except Exception as e:
            print_status("Bucket connection", False, str(e))
            
    except ImportError as e:
        print_status("Dependencies", False, f"Missing: {e}")
    except Exception as e:
        print_status("Error", False, str(e))
    
    return results


def check_azure_openai_configuration() -> dict[str, bool]:
    """Check Azure OpenAI configuration and connectivity."""
    print_section("Azure OpenAI")

    results = {
        "configured": False,
        "chat_connected": False,
        "embedding_connected": False,
    }

    try:
        from medical_agent.core.config import settings
        from medical_agent.infrastructure.azure_openai import (
            get_llama_index_embed_model,
        )

        # Check configuration
        if not settings.is_azure_openai_configured():
            print_status(
                "Configuration", False,
                "AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set"
            )
            return results

        print_status("Configuration", True)
        print(f"       Endpoint: {settings.azure_openai_endpoint}")
        print(f"       Chat deployment: {settings.azure_openai_deployment_name}")
        print(f"       Embedding deployment: {settings.azure_openai_embedding_deployment_name}")
        results["configured"] = True

        # Test LLM (LangChain)
        try:
            from langchain_openai import AzureChatOpenAI

            llm = AzureChatOpenAI(
                deployment_name=settings.azure_openai_deployment_name,
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
            print_status("LLM client (LangChain)", True)
            results["chat_connected"] = True
        except Exception as e:
            print_status("LLM client (LangChain)", False, str(e))

        # Test embedding model (LlamaIndex)
        try:
            embed_model = get_llama_index_embed_model()
            print_status("Embedding model (LlamaIndex)", True)
            results["embedding_connected"] = True
        except Exception as e:
            print_status("Embedding model (LlamaIndex)", False, str(e))

    except ImportError as e:
        print_status("Dependencies", False, f"Missing: {e}")
    except Exception as e:
        print_status("Error", False, str(e))

    return results


def check_langsmith_configuration() -> dict[str, bool]:
    """Check LangSmith configuration and connectivity."""
    print_section("LangSmith Observability (Optional)")

    results = {"configured": False}

    try:
        from medical_agent.core.config import settings

        # Check configuration
        if settings.is_langsmith_configured():
            print_status("Configuration", True)
            print(f"       Project: {settings.langsmith_project}")
            results["configured"] = True
        else:
            print_status(
                "Configuration", False,
                "LANGSMITH_API_KEY or LANGSMITH_TRACING not set (optional)"
            )
            return results

    except Exception as e:
        print_status("Error", False, str(e))

    return results


def check_database_configuration() -> dict[str, bool]:
    """Check PostgreSQL database configuration."""
    print_section("PostgreSQL + pgvector")
    
    results = {"configured": False, "connected": False, "pgvector": False}
    
    try:
        from medical_agent.core.config import settings
        
        # Check configuration
        print_status("Configuration", True)
        print(f"       Host: {settings.postgres_host}:{settings.postgres_port}")
        print(f"       Database: {settings.postgres_db}")
        results["configured"] = True
        
        # Try to connect
        try:
            import asyncio
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine
            
            async def test_connection() -> bool:
                engine = create_async_engine(
                    settings.database_connection_string,
                    echo=False,
                )
                async with engine.begin() as conn:
                    # Test basic connection
                    await conn.execute(text("SELECT 1"))
                    
                    # Check pgvector extension
                    result = await conn.execute(
                        text("SELECT * FROM pg_extension WHERE extname = 'vector'")
                    )
                    has_vector = result.fetchone() is not None
                    
                await engine.dispose()
                return has_vector
            
            has_vector = asyncio.run(test_connection())
            print_status("Connection", True)
            results["connected"] = True
            
            if has_vector:
                print_status("pgvector extension", True)
                results["pgvector"] = True
            else:
                print_status("pgvector extension", False, "Not installed")
                
        except Exception as e:
            print_status("Connection", False, str(e))
            
    except ImportError as e:
        print_status("Dependencies", False, f"Missing: {e}")
    except Exception as e:
        print_status("Error", False, str(e))
    
    return results


def print_setup_guide() -> None:
    """Print step-by-step setup instructions."""
    print_header("Infrastructure Setup Guide")
    
    print("""
This guide walks you through setting up all cloud services for the
FemTech Medical RAG Agent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: GCP Project Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create a GCP Project:
   → Go to https://console.cloud.google.com
   → Create new project or select existing one
   → Note your Project ID

2. Enable Required APIs:
   → Cloud Storage API
   → Cloud SQL Admin API

3. Create a Cloud Storage Bucket:
   → Go to Cloud Storage → Buckets
   → Create bucket: "phera_researchpaper"
   → Region: Choose closest to your users
   → Storage class: Standard
   → Access control: Uniform

4. Create a Service Account:
   → Go to IAM & Admin → Service Accounts
   → Create service account: "phera-backend"
   → Grant roles:
     • Storage Object Admin
     • Cloud SQL Client
   → Create JSON key and download

5. Set Environment Variables:
   GCP_PROJECT_ID=your-project-id
   GCP_BUCKET_NAME=phera_researchpaper
   GCP_CREDENTIALS_PATH=/path/to/service-account.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: GCP Cloud SQL (Production Database)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create Cloud SQL Instance:
   → Go to Cloud SQL → Create Instance
   → Choose PostgreSQL 16
   → Instance ID: "phera-pg-main"
   → Set password for 'postgres' user
   → Region: Same as your bucket
   → Machine type: db-f1-micro (dev) or larger

2. Configure Instance:
   → Enable public IP (or private for production)
   → Add authorized networks for your IP
   → Create database: "phera_postgres"

3. Install pgvector Extension:
   → Connect to instance via Cloud Shell or psql
   → Run: CREATE EXTENSION IF NOT EXISTS vector;

4. Set Environment Variables (production):
   DATABASE_URL=postgresql+asyncpg://USER:PASS@/phera_postgres?host=/cloudsql/PROJECT:REGION:INSTANCE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: Azure OpenAI Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create Azure OpenAI Resource:
   → Go to https://portal.azure.com
   → Create resource → Azure OpenAI
   → Region: Choose with GPT-4o availability
   → Pricing tier: Standard S0

2. Deploy Models in Azure OpenAI Studio:
   → Go to Azure OpenAI Studio
   → Deployments → Create deployment
   
   Chat Model:
   → Model: gpt-4o
   → Deployment name: gpt-4o
   → TPM limit: 30K+ recommended
   
   Embedding Model:
   → Model: text-embedding-3-small
   → Deployment name: text-embedding-3-small
   → TPM limit: 100K+ recommended

3. Get Credentials:
   → Keys and Endpoint in Azure Portal
   → Copy Endpoint and Key 1

4. Set Environment Variables:
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4: LlamaParser (LlamaCloud) Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create LlamaCloud Account:
   → Go to https://cloud.llamaindex.ai
   → Sign up for an account (free tier available)

2. Get API Key:
   → Go to API Keys section
   → Create a new API key
   → Copy the key

3. Set Environment Variables:
   LLAMA_CLOUD_API_KEY=llx-your-api-key

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5: Local Development Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Copy environment template:
   cp .env.example .env

2. Fill in all values in .env

3. Start local database:
   docker-compose up -d postgres

4. Run migrations:
   alembic upgrade head

5. Verify setup:
   python scripts/setup_infrastructure.py --check-all

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def check_all() -> bool:
    """Run all infrastructure checks."""
    print_header("FemTech Medical RAG Agent - Infrastructure Check")
    
    all_passed = True
    
    # Check .env file first
    if not check_env_file():
        all_passed = False
    
    # Check database
    db_results = check_database_configuration()
    if not db_results["connected"]:
        all_passed = False
    
    # Check GCP
    gcp_results = check_gcp_configuration()
    if not gcp_results["connected"]:
        all_passed = False
    
    # Check Azure OpenAI
    azure_openai_results = check_azure_openai_configuration()
    if not azure_openai_results["chat_connected"]:
        all_passed = False

    # Check LangSmith (optional)
    check_langsmith_configuration()
    
    # Summary
    print_header("Summary")
    
    if all_passed:
        print("\n  ✅ All infrastructure checks passed!")
        print("     Your environment is ready for development.\n")
    else:
        print("\n  ⚠️  Some checks failed.")
        print("     Run with --setup-guide for setup instructions.\n")
    
    return all_passed


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Infrastructure setup and verification for FemTech Medical RAG Agent"
    )
    
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Run all infrastructure checks",
    )
    parser.add_argument(
        "--check",
        choices=["gcp", "azure", "azure-openai", "langsmith", "database"],
        help="Check specific service",
    )
    parser.add_argument(
        "--setup-guide",
        action="store_true",
        help="Print setup instructions",
    )
    
    args = parser.parse_args()
    
    if args.setup_guide:
        print_setup_guide()
    elif args.check_all:
        success = check_all()
        sys.exit(0 if success else 1)
    elif args.check:
        if args.check == "gcp":
            check_gcp_configuration()
        elif args.check in ("azure", "azure-openai"):
            check_azure_openai_configuration()
        elif args.check == "langsmith":
            check_langsmith_configuration()
        elif args.check == "database":
            check_database_configuration()
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()

