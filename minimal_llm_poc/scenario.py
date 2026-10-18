"""Scenario definitions for the Minimal LLM PoC.

Transfers the 'Dependency Change' scenario from the controlled benchmark
to realistic LLM agent tasks.  Three architectural phases of 2 episodes each
(6 episodes total), with explicit dependency (synergy) shifts between phases.

Architectural Slots:
  Slot 0: auth          - User authentication, OAuth, JWT, sessions
  Slot 1: user_db       - Relational user store, migrations, indexing
  Slot 2: cart          - Shopping cart, checkout, line items
  Slot 3: inventory     - Stock availability, reservation, fulfillment
  Slot 4: payment       - Payment processing, webhooks, idempotency
  Slot 5: order         - Order lifecycle, state machine, fulfillment
  Slot 6: routing       - API gateway, service routing, event broker
  Slot 7: observability - Structured logging, rate limits, alerting

Topological Dependency Evolution:
  Phase 1 (Episodes 0-1, Monolith Split):
    Synergies: (0, 1), (2, 3), (4, 5)
  Phase 2 (Episodes 2-3, Gateway & Direct Checkout):
    Synergies: (0, 6), (2, 4), (3, 5)
  Phase 3 (Episodes 4-5, Event-Driven Platform):
    Synergies: (0, 2), (3, 6), (5, 6), (4, 7)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Agent definitions

@dataclass(frozen=True)
class AgentDef:
    """Defines an LLM agent with a role, model, and system prompt."""
    id: str
    role: str
    model: str
    system_prompt: str
    # 4D Capability vector: [backend/db, infra/devops, security/auth, api/contract]
    capability: Tuple[float, float, float, float]


AGENTS: List[AgentDef] = [
    AgentDef(
        id="backend",
        role="Backend Systems Specialist",
        model="gemini-3.5-flash-lite",
        system_prompt=(
            "You are a backend systems specialist. You are an expert in "
            "server-side application code, relational database schemas, SQL "
            "queries, state machines, and business logic implementation. "
            "You write clean, production-ready code. When given a task, "
            "produce ONLY the requested output (code, config, or schema) "
            "with no explanation unless explicitly asked."
        ),
        capability=(0.9, 0.2, 0.3, 0.4),
    ),
    AgentDef(
        id="infra",
        role="Infrastructure & DevOps Engineer",
        model="gemini-3.1-flash-lite",
        system_prompt=(
            "You are an infrastructure and DevOps engineer. You are an expert "
            "in Docker, Kubernetes manifests, CI/CD pipelines, networking, "
            "service mesh configuration, event bus infrastructure, and "
            "monitoring/alerting setup. When given a task, produce ONLY the "
            "requested output (config, YAML, or script) with no explanation "
            "unless explicitly asked."
        ),
        capability=(0.2, 0.9, 0.2, 0.3),
    ),
    AgentDef(
        id="security",
        role="Security & Auth Specialist",
        model="gemma-4-26b-a4b-it",
        system_prompt=(
            "You are a security and authentication specialist. You are an "
            "expert in OAuth2, JWT tokens, session management, access control "
            "lists, webhook signature verification, encryption, and secure "
            "API design. When given a task, produce ONLY the requested output "
            "(code, config, or specification) with no explanation unless "
            "explicitly asked."
        ),
        capability=(0.3, 0.2, 0.9, 0.3),
    ),
    AgentDef(
        id="frontend",
        role="Integration & API Specialist",
        model="gemini-3.5-flash-lite",
        system_prompt=(
            "You are an API integration specialist. You are an expert in "
            "REST API contract design, JSON/Avro schema definition, request/"
            "response validation, data serialization, and client-server "
            "protocol design. When given a task, produce ONLY the requested "
            "output (schema, spec, or code) with no explanation unless "
            "explicitly asked."
        ),
        capability=(0.3, 0.3, 0.3, 0.9),
    ),
]

AGENT_INDEX = {a.id: i for i, a in enumerate(AGENTS)}
NUM_AGENTS = len(AGENTS)

# Judge configuration (disjoint from agent models)

JUDGE_MODEL = "gemini-3.8-flash"
JUDGE_SYSTEM_PROMPT = (
    "You are a strict code quality evaluator. You assess whether an LLM output "
    "correctly implements the stated task requirements. "
    "Reply with ONLY a JSON object: {\"score\": 0, \"reason\": \"...\"} or "
    "{\"score\": 1, \"reason\": \"...\"}. "
    "Score 1 means the code or specification fully meets all stated requirements. "
    "Score 0 means it has significant omissions, syntax errors, or failing logic."
)

# Task definition schema

@dataclass(frozen=True)
class TaskDef:
    """Defines a single task for an LLM agent."""
    id: str
    phase: int              # 0, 1, or 2
    slot_index: int         # 0..7 — subsystem slot
    title: str
    prompt: str             # Self-contained prompt sent to the agent
    evaluation_mode: str    # "deterministic" or "llm_judge"
    required_patterns: Tuple[str, ...] = ()
    forbidden_patterns: Tuple[str, ...] = ()
    # 4D affinity vector: [backend/db, infra/devops, security/auth, api/contract]
    task_affinity: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)


# Slot Affinities (Common baseline per subsystem slot)
SLOT_AFFINITIES: Dict[int, Tuple[float, float, float, float]] = {
    0: (0.3, 0.2, 0.9, 0.3),  # auth -> security
    1: (0.9, 0.1, 0.3, 0.2),  # user_db -> backend
    2: (0.4, 0.2, 0.2, 0.8),  # cart -> frontend/api
    3: (0.7, 0.3, 0.1, 0.5),  # inventory -> backend
    4: (0.5, 0.2, 0.7, 0.4),  # payment -> security/backend
    5: (0.8, 0.2, 0.2, 0.4),  # order -> backend
    6: (0.2, 0.9, 0.3, 0.5),  # routing -> infra
    7: (0.3, 0.8, 0.2, 0.3),  # observability -> infra
}

# Phase 1: "Monolith Split" (Episodes 0-1)
# Synergies: (0, 1) [auth<->user_db], (2, 3) [cart<->inventory], (4, 5) [payment<->order]
PHASE1_TASKS: List[TaskDef] = [
    TaskDef(
        id="p1_s0_auth_schema",
        phase=0, slot_index=0,
        title="Design OAuth2 Token Storage Schema",
        prompt=(
            "Design a PostgreSQL CREATE TABLE statement for storing OAuth2 tokens.\n\n"
            "Requirements:\n"
            "- Table name: oauth_tokens\n"
            "- Columns: id (UUID, primary key, default gen_random_uuid()), "
            "user_id (UUID, NOT NULL, foreign key to users.id), "
            "access_token (TEXT, NOT NULL, UNIQUE), "
            "refresh_token (TEXT, UNIQUE), "
            "token_type (VARCHAR(20), NOT NULL, DEFAULT 'Bearer'), "
            "scope (TEXT), "
            "expires_at (TIMESTAMPTZ, NOT NULL), "
            "created_at (TIMESTAMPTZ, NOT NULL, DEFAULT NOW())\n"
            "- Add an index on user_id\n"
            "- Add an index on expires_at for cleanup queries\n\n"
            "Output ONLY the SQL statement(s), no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "CREATE TABLE", "oauth_tokens", "user_id", "access_token",
            "refresh_token", "token_type", "expires_at", "created_at",
            "UUID", "PRIMARY KEY", "NOT NULL",
        ),
        task_affinity=SLOT_AFFINITIES[0],
    ),
    TaskDef(
        id="p1_s1_user_db_migration",
        phase=0, slot_index=1,
        title="Write User Table Migration Script",
        prompt=(
            "Write an SQL migration script that:\n\n"
            "1. Adds an 'email_verified' BOOLEAN column (DEFAULT FALSE) to the 'users' table\n"
            "2. Adds a 'last_login_at' TIMESTAMPTZ column to the 'users' table\n"
            "3. Creates a partial index on users WHERE email_verified = FALSE\n"
            "4. Wraps everything in a transaction (BEGIN/COMMIT)\n\n"
            "Output ONLY the SQL migration, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "ALTER TABLE", "users", "email_verified", "BOOLEAN",
            "last_login_at", "TIMESTAMPTZ", "INDEX", "BEGIN", "COMMIT",
        ),
        task_affinity=SLOT_AFFINITIES[1],
    ),
    TaskDef(
        id="p1_s2_cart_api_spec",
        phase=0, slot_index=2,
        title="Design Cart REST API Contract",
        prompt=(
            "Define a REST API contract for a shopping cart service as a JSON object.\n\n"
            "Requirements:\n"
            "- Endpoint: POST /api/v1/cart/items\n"
            "- Request body: {\"product_id\": \"string\", \"quantity\": integer (min 1, max 99)}\n"
            "- Success response (200): {\"cart_id\": \"string\", \"items\": [{\"product_id\": \"string\", "
            "\"quantity\": integer, \"unit_price\": number}], \"total\": number}\n"
            "- Error response (400): {\"error\": \"string\", \"code\": \"string\"}\n"
            "- Error response (404): {\"error\": \"Product not found\", \"code\": \"PRODUCT_NOT_FOUND\"}\n\n"
            "Output ONLY the JSON API specification object, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "product_id", "quantity", "cart_id", "items", "total",
            "unit_price", "error", "code", "PRODUCT_NOT_FOUND",
        ),
        task_affinity=SLOT_AFFINITIES[2],
    ),
    TaskDef(
        id="p1_s3_inventory_check",
        phase=0, slot_index=3,
        title="Implement Inventory Availability Check",
        prompt=(
            "Write a Python function `check_availability(inventory: dict, requested_qty: int) -> dict` that:\n\n"
            "- Input: inventory = {\"item_id\": str, \"stock\": int, \"reserved\": int}\n"
            "- Computes unreserved = stock - reserved\n"
            "- Returns {\"available\": bool, \"unreserved\": int, \"can_fulfill\": bool}\n"
            "- available is True if unreserved > 0\n"
            "- can_fulfill is True if unreserved >= requested_qty\n"
            "- If stock < 0 or reserved < 0, raise ValueError(\"Invalid inventory state\")\n\n"
            "Output ONLY the Python function, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "def check_availability", "unreserved", "available",
            "can_fulfill", "ValueError", "Invalid inventory state",
        ),
        task_affinity=SLOT_AFFINITIES[3],
    ),
    TaskDef(
        id="p1_s4_payment_webhook",
        phase=0, slot_index=4,
        title="Configure Payment Gateway Webhook Handler",
        prompt=(
            "Write a Python function `handle_payment_webhook(payload: dict, signature: str, secret: str) -> dict` that:\n\n"
            "1. Verifies the webhook signature using HMAC-SHA256: hmac.new(secret.encode(), json.dumps(payload, sort_keys=True).encode(), hashlib.sha256).hexdigest()\n"
            "2. If signature is invalid, returns {\"status\": \"rejected\", \"reason\": \"invalid_signature\"}\n"
            "3. Validates payload has fields: 'event_type', 'payment_id', 'amount', 'currency'\n"
            "4. If event_type is 'payment.completed', returns {\"status\": \"accepted\", \"action\": \"fulfill_order\", \"payment_id\": payload['payment_id']}\n"
            "5. If event_type is 'payment.failed', returns {\"status\": \"accepted\", \"action\": \"cancel_order\", \"payment_id\": payload['payment_id']}\n"
            "6. For unknown event types, returns {\"status\": \"accepted\", \"action\": \"ignore\"}\n\n"
            "Output ONLY the Python function with necessary imports, no explanation."
        ),
        evaluation_mode="llm_judge",
        task_affinity=SLOT_AFFINITIES[4],
    ),
    TaskDef(
        id="p1_s5_order_status_machine",
        phase=0, slot_index=5,
        title="Implement Order Status State Machine",
        prompt=(
            "Write a Python class `OrderStateMachine` that:\n\n"
            "- Has states: 'created', 'paid', 'processing', 'shipped', 'delivered', 'cancelled'\n"
            "- Valid transitions:\n"
            "  created -> paid, created -> cancelled\n"
            "  paid -> processing, paid -> cancelled\n"
            "  processing -> shipped\n"
            "  shipped -> delivered\n"
            "- __init__(self) sets self.state = 'created'\n"
            "- transition(self, new_state) -> str: returns new state if valid, raises ValueError('Invalid transition from {current} to {new}') if not\n"
            "- get_state(self) -> str: returns current state\n\n"
            "Output ONLY the Python class, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "class OrderStateMachine", "created", "paid", "processing",
            "shipped", "delivered", "cancelled", "transition",
            "ValueError", "Invalid transition",
        ),
        task_affinity=SLOT_AFFINITIES[5],
    ),
    TaskDef(
        id="p1_s6_service_health",
        phase=0, slot_index=6,
        title="Create Service Health Probe Endpoint",
        prompt=(
            "Write a Python Flask function `health_check()` that:\n\n"
            "- Route: GET /health\n"
            "- Returns a JSON response with status 200:\n"
            "  {\"status\": \"healthy\", \"service\": \"core-monolith\", \"version\": \"1.0.0\"}\n"
            "- Includes header 'X-Service-Status': 'ready'\n\n"
            "Output ONLY the Python function with decorator, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "health_check", "/health", "GET", "healthy",
            "core-monolith", "1.0.0", "X-Service-Status", "ready",
        ),
        task_affinity=SLOT_AFFINITIES[6],
    ),
    TaskDef(
        id="p1_s7_logging_config",
        phase=0, slot_index=7,
        title="Configure Structured JSON Logging",
        prompt=(
            "Write a Python dictionary `LOGGING_CONFIG` that configures Python's "
            "logging module for structured JSON output.\n\n"
            "Requirements:\n"
            "- Use dictConfig format (version: 1)\n"
            "- One formatter named 'json' with format string containing timestamp, level, logger, message\n"
            "- One handler named 'console' using StreamHandler with the json formatter\n"
            "- Root logger level: INFO, handlers: ['console']\n"
            "- disable_existing_loggers: False\n\n"
            "Output ONLY the Python dictionary assignment, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "LOGGING_CONFIG", "version", "formatters", "json",
            "handlers", "console", "StreamHandler", "INFO",
            "disable_existing_loggers", "False",
        ),
        task_affinity=SLOT_AFFINITIES[7],
    ),
]

# Phase 2: "Gateway & Microservice Integration" (Episodes 2-3)
# Synergies: (0, 6) [auth<->routing], (2, 4) [cart<->payment], (3, 5) [inventory<->order]
PHASE2_TASKS: List[TaskDef] = [
    TaskDef(
        id="p2_s0_auth_jwt_middleware",
        phase=1, slot_index=0,
        title="Implement Gateway JWT Validation Middleware",
        prompt=(
            "Write a Python middleware function `validate_jwt(token: str, public_key: str) -> dict` that:\n\n"
            "1. Splits the token into header, payload, signature (separated by '.')\n"
            "2. If token doesn't have exactly 3 parts, raises ValueError('Malformed token')\n"
            "3. Decodes the payload from base64url encoding\n"
            "4. Parses the decoded payload as JSON\n"
            "5. Checks 'exp' field: if present and < current Unix timestamp, raises ValueError('Token expired')\n"
            "6. Checks 'iss' field: if present and != 'auth-service', raises ValueError('Invalid issuer')\n"
            "7. Returns the decoded payload dict\n\n"
            "Output ONLY the Python function with imports, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "def validate_jwt", "base64", "Malformed token",
            "Token expired", "Invalid issuer", "auth-service",
            "exp", "iss",
        ),
        task_affinity=SLOT_AFFINITIES[0],
    ),
    TaskDef(
        id="p2_s1_user_db_indexes",
        phase=1, slot_index=1,
        title="PostgreSQL Index Optimization for Microservices",
        prompt=(
            "Write SQL statements to optimize token and user lookups across microservices:\n\n"
            "1. Create a B-Tree index on `oauth_tokens(access_token)`\n"
            "2. Create a compound index on `users(email, tenant_id)`\n"
            "3. Create a partial index on `oauth_tokens(user_id)` WHERE expires_at > NOW()\n\n"
            "Output ONLY the SQL index statements, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "CREATE INDEX", "oauth_tokens", "access_token",
            "users", "email", "tenant_id", "expires_at > NOW()",
        ),
        task_affinity=SLOT_AFFINITIES[1],
    ),
    TaskDef(
        id="p2_s2_cart_checkout",
        phase=1, slot_index=2,
        title="Implement Cart Checkout Coordination",
        prompt=(
            "Write a Python async function `checkout(cart_id: str, payment_method: str) -> dict` that:\n\n"
            "1. Validates cart_id is not empty, raises ValueError('Empty cart_id') otherwise\n"
            "2. Validates payment_method is one of ['credit_card', 'paypal', 'bank_transfer'], "
            "raises ValueError('Unsupported payment method') otherwise\n"
            "3. Creates an order dict with: order_id (uuid4), cart_id, payment_method, "
            "status='pending', created_at (ISO format)\n"
            "4. Returns {\"order_id\": str, \"status\": \"pending\", \"payment_url\": f\"https://pay.example.com/{order_id}\"}\n"
            "Handle all exceptions and return {\"error\": str(e), \"status\": \"failed\"} on failure.\n\n"
            "Output ONLY the Python function with imports, no explanation."
        ),
        evaluation_mode="llm_judge",
        task_affinity=SLOT_AFFINITIES[2],
    ),
    TaskDef(
        id="p2_s3_inventory_reserve",
        phase=1, slot_index=3,
        title="Implement Inventory Reservation with TTL",
        prompt=(
            "Write a Python class `InventoryReserver` that:\n\n"
            "- __init__(self, ttl_seconds: int = 900)\n"
            "- Has self._reservations: Dict[str, dict]\n"
            "- reserve(self, item_id: str, quantity: int) -> dict:\n"
            "  Returns {\"reservation_id\": str(uuid4), \"item_id\": item_id, "
            "\"quantity\": quantity, \"expires_at\": ISO timestamp, \"status\": \"reserved\"}\n"
            "- release(self, reservation_id: str) -> dict:\n"
            "  Removes reservation, returns {\"reservation_id\": reservation_id, \"status\": \"released\"}\n"
            "  Raises KeyError if not found\n"
            "- is_valid(self, reservation_id: str) -> bool:\n"
            "  Returns True if reservation exists and hasn't expired\n\n"
            "Output ONLY the Python class with imports, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "class InventoryReserver", "ttl_seconds", "_reservations",
            "reserve", "release", "is_valid", "reservation_id",
            "expires_at",
        ),
        task_affinity=SLOT_AFFINITIES[3],
    ),
    TaskDef(
        id="p2_s4_payment_idempotency",
        phase=1, slot_index=4,
        title="Design Payment Retry Handler with Idempotency",
        prompt=(
            "Write a Python class `PaymentRetryHandler` that:\n\n"
            "- __init__(self, max_retries: int = 3, base_delay: float = 1.0)\n"
            "- Has a dict self._idempotency_store: Dict[str, dict] for tracking processed payments\n"
            "- execute(self, payment_id: str, amount: float, callback) -> dict:\n"
            "  1. If payment_id in _idempotency_store, return the stored result\n"
            "  2. Retry loop: attempt callback(payment_id, amount) up to max_retries times\n"
            "  3. On success, store result in _idempotency_store and return it\n"
            "  4. Delay between retries: base_delay * (2 ** attempt)\n"
            "  5. On exhaustion, return {\"status\": \"failed\", \"payment_id\": payment_id, \"retries_exhausted\": True}\n\n"
            "Output ONLY the Python class, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "class PaymentRetryHandler", "max_retries", "base_delay",
            "_idempotency_store", "execute", "retries_exhausted",
            "2 **",
        ),
        task_affinity=SLOT_AFFINITIES[4],
    ),
    TaskDef(
        id="p2_s5_order_fulfillment",
        phase=1, slot_index=5,
        title="Coordinate Order Fulfillment with Reservation",
        prompt=(
            "Write a Python function `fulfill_order(order: dict, reserver) -> dict` that:\n\n"
            "1. Validates order status is 'paid', raises ValueError('Order not paid') if not\n"
            "2. For each item in order['items'], calls reserver.reserve(item['id'], item['quantity'])\n"
            "3. If any reservation fails or is invalid, rolls back previous reservations via reserver.release and raises RuntimeError('Reservation failed')\n"
            "4. On all successful reservations, returns {\"order_id\": order['id'], \"status\": \"fulfilled\", \"reservation_ids\": list_of_ids}\n\n"
            "Output ONLY the Python function with necessary error handling, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "def fulfill_order", "Order not paid", "reserver.reserve",
            "reserver.release", "Reservation failed", "fulfilled",
            "reservation_ids",
        ),
        task_affinity=SLOT_AFFINITIES[5],
    ),
    TaskDef(
        id="p2_s6_gateway_routing",
        phase=1, slot_index=6,
        title="Configure API Gateway Routing Rules",
        prompt=(
            "Write an API gateway routing configuration in YAML format.\n\n"
            "Requirements:\n"
            "- Route /api/v1/auth/** to service 'auth-service' on port 8081\n"
            "- Route /api/v1/cart/** to service 'cart-service' on port 8082\n"
            "- Route /api/v1/orders/** to service 'order-service' on port 8083\n"
            "- Each route must have:\n"
            "  - path_prefix matching\n"
            "  - strip_prefix: true\n"
            "  - timeout_ms: 5000\n"
            "  - max_retries: 2\n"
            "- Global CORS config: allow origins ['https://app.example.com'], "
            "allow methods ['GET', 'POST', 'PUT', 'DELETE']\n\n"
            "Output ONLY the YAML configuration, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "auth-service", "cart-service", "order-service",
            "8081", "8082", "8083", "strip_prefix", "timeout_ms",
            "max_retries", "CORS", "app.example.com",
        ),
        task_affinity=SLOT_AFFINITIES[6],
    ),
    TaskDef(
        id="p2_s7_rate_limit_policy",
        phase=1, slot_index=7,
        title="Configure API Rate Limiting Policy",
        prompt=(
            "Write an API rate limiting configuration in YAML format.\n\n"
            "Requirements:\n"
            "- Global rate limit: 1000 requests per minute\n"
            "- Endpoint-specific limits:\n"
            "  /api/v1/auth/login: 10 requests per minute per IP\n"
            "  /api/v1/cart/checkout: 20 requests per minute per IP\n"
            "- Response on limit exceeded: HTTP 429 with header 'Retry-After'\n"
            "- Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset\n\n"
            "Output ONLY the YAML configuration, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "1000", "login", "10", "checkout", "20", "429",
            "Retry-After", "X-RateLimit",
        ),
        task_affinity=SLOT_AFFINITIES[7],
    ),
]

# Phase 3: "Event-Driven Reactive Platform" (Episodes 4-5)
# Synergies: (0, 2) [auth<->cart], (3, 6) [inventory<->routing], (5, 6) [order<->routing], (4, 7) [payment<->observability]
PHASE3_TASKS: List[TaskDef] = [
    TaskDef(
        id="p3_s0_auth_session_store",
        phase=2, slot_index=0,
        title="Implement Distributed Session Store",
        prompt=(
            "Write a Python class `DistributedSessionStore` that:\n\n"
            "- __init__(self, ttl_seconds: int = 3600)\n"
            "- Has self._sessions: Dict[str, dict]\n"
            "- create_session(self, user_id: str, initial_data: dict = None) -> str:\n"
            "  Creates session with session_id (uuid4), user_id, data (dict), "
            "expires_at (now + ttl)\n"
            "- get_session(self, session_id: str) -> dict | None:\n"
            "  Returns session data if valid and unexpired; returns None if expired or not found\n"
            "- sync_cart_data(self, session_id: str, cart_data: dict) -> bool:\n"
            "  Updates session['data']['cart'] = cart_data; returns True if session exists, False otherwise\n\n"
            "Output ONLY the Python class with imports, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "class DistributedSessionStore", "ttl_seconds", "_sessions",
            "create_session", "get_session", "sync_cart_data",
            "expires_at", "uuid",
        ),
        task_affinity=SLOT_AFFINITIES[0],
    ),
    TaskDef(
        id="p3_s1_user_event_stream",
        phase=2, slot_index=1,
        title="User Event Streaming Specification",
        prompt=(
            "Design a JSON Schema (draft 2020-12) for a `UserProfileUpdated` event:\n\n"
            "- type: object\n"
            "- required fields: event_id (uuid), event_type ('user.profile_updated'), "
            "timestamp (date-time), user_id (uuid), changed_fields (array of strings)\n"
            "- additionalProperties: false\n\n"
            "Output ONLY the JSON schema document, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "event_id", "event_type", "user.profile_updated",
            "timestamp", "user_id", "changed_fields", "uuid",
            "date-time", "additionalProperties",
        ),
        task_affinity=SLOT_AFFINITIES[1],
    ),
    TaskDef(
        id="p3_s2_cart_session_sync",
        phase=2, slot_index=2,
        title="Sync Cart State to Distributed Session",
        prompt=(
            "Write a Python function `sync_cart_to_session(session_store, session_id: str, cart: dict) -> dict` that:\n\n"
            "1. Validates session_id is valid via session_store.get_session(session_id)\n"
            "2. If invalid, raises ValueError('Session expired or not found')\n"
            "3. Validates cart has 'items' (list) and 'total' (float >= 0)\n"
            "4. Calls session_store.sync_cart_data(session_id, cart)\n"
            "5. Returns {\"status\": \"synced\", \"session_id\": session_id, \"item_count\": len(cart['items'])}\n\n"
            "Output ONLY the Python function with error handling, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "def sync_cart_to_session", "session_store.get_session",
            "Session expired or not found", "items", "total",
            "sync_cart_data", "synced", "item_count",
        ),
        task_affinity=SLOT_AFFINITIES[2],
    ),
    TaskDef(
        id="p3_s3_inventory_event_consumer",
        phase=2, slot_index=3,
        title="Implement Event Consumer for Inventory Updates",
        prompt=(
            "Write a Python class `InventoryEventConsumer` that:\n\n"
            "- __init__(self, stock_store: dict)\n"
            "- process_event(self, event: dict) -> dict:\n"
            "  1. Validates event has 'event_type', 'item_id', 'quantity'\n"
            "  2. If event_type == 'order.placed': subtract quantity from stock_store\n"
            "  3. If event_type == 'order.cancelled': add quantity back to stock_store\n"
            "  4. If resulting stock < 0, raise ValueError('Insufficient inventory')\n"
            "  5. Returns {\"item_id\": event['item_id'], \"new_stock\": stock_store[event['item_id']], \"processed\": True}\n\n"
            "Output ONLY the Python class, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "class InventoryEventConsumer", "process_event",
            "order.placed", "order.cancelled", "Insufficient inventory",
            "new_stock", "processed",
        ),
        task_affinity=SLOT_AFFINITIES[3],
    ),
    TaskDef(
        id="p3_s4_payment_dlq_alerting",
        phase=2, slot_index=4,
        title="Payment DLQ and Anomaly Dispatcher",
        prompt=(
            "Write a Python class `PaymentDLQDispatcher` that:\n\n"
            "- __init__(self, alert_sink)\n"
            "- Has self.dlq: List[dict] = []\n"
            "- dispatch_failure(self, payment_id: str, error: str, payload: dict) -> dict:\n"
            "  1. Formats record: {\"payment_id\": payment_id, \"error\": error, \"timestamp\": ISO, \"payload\": payload}\n"
            "  2. Appends record to self.dlq\n"
            "  3. Calls alert_sink.notify_payment_alert(payment_id, error)\n"
            "  4. Returns {\"status\": \"dead_lettered\", \"payment_id\": payment_id, \"dlq_size\": len(self.dlq)}\n\n"
            "Output ONLY the Python class with imports, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "class PaymentDLQDispatcher", "dlq", "dispatch_failure",
            "notify_payment_alert", "dead_lettered", "dlq_size",
        ),
        task_affinity=SLOT_AFFINITIES[4],
    ),
    TaskDef(
        id="p3_s5_order_event_publisher",
        phase=2, slot_index=5,
        title="Publish Order State Changes to Event Broker",
        prompt=(
            "Write a Python class `OrderEventPublisher` that:\n\n"
            "- __init__(self, broker)\n"
            "- publish_state_change(self, order_id: str, old_state: str, new_state: str) -> dict:\n"
            "  1. Constructs event: {\"event_id\": str(uuid4), \"topic\": \"order.events\", "
            "\"order_id\": order_id, \"old_state\": old_state, \"new_state\": new_state, \"timestamp\": ISO}\n"
            "  2. Calls broker.publish(\"order.events\", event)\n"
            "  3. Returns event\n\n"
            "Output ONLY the Python class with imports, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "class OrderEventPublisher", "publish_state_change",
            "order.events", "broker.publish", "old_state", "new_state",
            "uuid",
        ),
        task_affinity=SLOT_AFFINITIES[5],
    ),
    TaskDef(
        id="p3_s6_event_broker_config",
        phase=2, slot_index=6,
        title="Configure Event Bus Broker Topics and Partitioning",
        prompt=(
            "Write an Event Broker configuration in YAML format for Apache Kafka or RabbitMQ:\n\n"
            "Requirements:\n"
            "- Topic 'order.events': 6 partitions, replication factor 3, cleanup policy 'delete', retention 7 days\n"
            "- Topic 'inventory.events': 4 partitions, replication factor 3, retention 3 days\n"
            "- Dead-letter topic 'payments.dlq': 1 partition, replication factor 3, retention 30 days\n"
            "- Global consumer group 'inventory-workers': max poll interval 300000ms\n\n"
            "Output ONLY the YAML configuration, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "order.events", "inventory.events", "payments.dlq",
            "partitions", "replication", "retention", "inventory-workers",
        ),
        task_affinity=SLOT_AFFINITIES[6],
    ),
    TaskDef(
        id="p3_s7_observability_alerting",
        phase=2, slot_index=7,
        title="Configure Prometheus Alerting Rules for Payment DLQ",
        prompt=(
            "Write Prometheus alerting rules in YAML format:\n\n"
            "Requirements:\n"
            "- Rule group: 'payment-alerts'\n"
            "- Alert 'HighPaymentFailureRate':\n"
            "  - expr: rate(payment_dlq_total[5m]) > 0.05\n"
            "  - for: 2m\n"
            "  - severity: critical\n"
            "  - description: 'Payment DLQ failure rate exceeds 5% in the last 5 minutes'\n"
            "- Alert 'InventoryLowStock':\n"
            "  - expr: inventory_available_stock < 5\n"
            "  - for: 5m\n"
            "  - severity: warning\n\n"
            "Output ONLY the YAML alert rule configuration, no explanation."
        ),
        evaluation_mode="deterministic",
        required_patterns=(
            "payment-alerts", "HighPaymentFailureRate", "payment_dlq_total",
            "critical", "InventoryLowStock", "inventory_available_stock",
            "warning",
        ),
        task_affinity=SLOT_AFFINITIES[7],
    ),
]

# Phase Aggregation and Horizon
ALL_PHASES: List[List[TaskDef]] = [PHASE1_TASKS, PHASE2_TASKS, PHASE3_TASKS]

NUM_EPISODES = 6
NUM_TASKS_PER_EPISODE = 8
NUM_PHASES = 3
EPISODES_PER_PHASE = NUM_EPISODES // NUM_PHASES  # 2 episodes per phase


def get_phase(episode: int) -> int:
    """Return phase index (0, 1, or 2) for a given episode number."""
    return min(episode // EPISODES_PER_PHASE, NUM_PHASES - 1)


def get_tasks_for_episode(episode: int) -> List[TaskDef]:
    """Return the 8 tasks active in the given episode."""
    return ALL_PHASES[get_phase(episode)]


# Explicit Topological Interaction Graphs (Synergies per Phase)
# Phase 1: (0, 1), (2, 3), (4, 5)
# Phase 2: (0, 6), (2, 4), (3, 5)
# Phase 3: (0, 2), (3, 6), (5, 6), (4, 7)

PHASE_SYNERGY_EDGES: Dict[int, List[Tuple[int, int]]] = {
    0: [(0, 1), (2, 3), (4, 5)],
    1: [(0, 6), (2, 4), (3, 5)],
    2: [(0, 2), (3, 6), (5, 6), (4, 7)],
}


def get_interaction_graph(episode: int) -> List[List[float]]:
    """Return the 8x8 interaction (synergy) matrix for the given episode.

    1.0 at (i,j) means tasks i and j are synergistic (benefit from co-assignment).
    """
    phase = get_phase(episode)
    edges = PHASE_SYNERGY_EDGES[phase]
    M = NUM_TASKS_PER_EPISODE
    graph = [[0.0] * M for _ in range(M)]
    for i, j in edges:
        graph[i][j] = 1.0
        graph[j][i] = 1.0
    return graph


def get_synergy_partners(episode: int, slot: int) -> List[int]:
    """Return list of slot indices that synergize with `slot` in this episode."""
    phase = get_phase(episode)
    edges = PHASE_SYNERGY_EDGES[phase]
    partners = []
    for i, j in edges:
        if i == slot:
            partners.append(j)
        elif j == slot:
            partners.append(i)
    return partners
