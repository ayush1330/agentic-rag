-- =============================================================================
-- FemTech Medical RAG Agent - Database Initialization
-- =============================================================================
-- This script runs automatically when the PostgreSQL container starts

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges (useful for development)
GRANT ALL PRIVILEGES ON DATABASE femtech_medical TO femtech;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'FemTech Medical database initialized successfully with pgvector extension';
END $$;


