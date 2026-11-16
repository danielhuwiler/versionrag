"""Async Neo4j client wrapper."""

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncResult

logger = logging.getLogger(__name__)

# DDL statements executed once at startup to ensure the graph schema is ready.
# All statements use IF NOT EXISTS so they are safe to run on every startup.
_SCHEMA_STATEMENTS: list[tuple[str, str]] = [
    (
        "category_unique",
        "CREATE CONSTRAINT category_unique IF NOT EXISTS "
        "FOR (c:Category) REQUIRE c.name IS UNIQUE",
    ),
    (
        "document_unique",
        "CREATE CONSTRAINT document_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE (d.name, d.category) IS UNIQUE",
    ),
    (
        "version_unique",
        "CREATE CONSTRAINT version_unique IF NOT EXISTS "
        "FOR (v:Version) REQUIRE (v.number, v.document) IS UNIQUE",
    ),
    (
        "change_type_idx",
        "CREATE INDEX change_type_idx IF NOT EXISTS "
        "FOR (c:Change) ON (c.change_type)",
    ),
    (
        "version_sort_key_idx",
        "CREATE INDEX version_sort_key_idx IF NOT EXISTS "
        "FOR (v:Version) ON (v.sort_key)",
    ),
    (
        "change_fulltext_idx",
        "CREATE FULLTEXT INDEX change_fulltext_idx IF NOT EXISTS "
        "FOR (c:Change) ON EACH [c.name, c.description]",
    ),
]


class Neo4jClient:
    """Manages the async Neo4j driver and exposes a typed query interface."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Create the async driver and verify connectivity."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        await self._driver.verify_connectivity()
        logger.info("Neo4j connection established: %s", self._uri)

    async def close(self) -> None:
        """Close the driver and release all connections."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j driver closed")

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dicts.

        Args:
            query: Cypher query string.
            parameters: Optional query parameters.
            database: Target database name.

        Returns:
            List of records as plain dictionaries.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self._driver:
            raise RuntimeError("Neo4j client is not connected. Call connect() first.")

        async with self._driver.session(database=database) as session:
            result: AsyncResult = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def ensure_schema(self) -> list[str]:
        """Create graph constraints and indexes if they do not already exist.

        Runs all DDL statements defined in ``_SCHEMA_STATEMENTS`` using
        ``IF NOT EXISTS``, so repeated startups are safe.

        Returns:
            List of constraint/index names that were ensured.

        Raises:
            RuntimeError: If the client is not connected.
        """
        names: list[str] = []
        for name, stmt in _SCHEMA_STATEMENTS:
            await self.execute_query(stmt)
            logger.debug("Schema ensured: %s", name)
            names.append(name)
        logger.info("Neo4j schema ready (%d constraints/indexes)", len(names))
        return names

    async def verify_connectivity(self) -> bool:
        """Check whether the Neo4j instance is reachable.

        Returns:
            True if reachable, False otherwise.
        """
        try:
            await self.execute_query("RETURN 1 AS ok")
            return True
        except Exception as exc:
            logger.error("Neo4j connectivity check failed: %s", exc)
            return False

    async def get_constraint_names(self) -> list[str]:
        """Query Neo4j for the names of all active constraints.

        Returns:
            Sorted list of constraint name strings.
        """
        rows = await self.execute_query("SHOW CONSTRAINTS YIELD name RETURN name")
        return sorted(row["name"] for row in rows)
