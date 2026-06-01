"""Pydantic API schemas for graph_rag.

These types document the contract with the React frontend. Field naming
matches the frontend's CLD schema (polarity/subsystem/description) rather
than the raw KG schema (sign/category) — the mapping happens in
recommender.py / api/routes_suggest.py at the API boundary.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Common ─────────────────────────────────────────────────────────────

class CanvasNode(BaseModel):
    """A node currently on the user's canvas (frontend schema)."""
    id: str
    label: str
    category: Optional[str] = None
    subsystem: Optional[str] = None
    description: Optional[str] = None


class CanvasEdge(BaseModel):
    """A directed signed edge currently on the user's canvas."""
    source: str
    target: str
    polarity: Literal["+", "-"]
    label: Optional[str] = None
    description: Optional[str] = None


class CanvasState(BaseModel):
    """Snapshot of the user's CLD that the backend reasons over."""
    nodes: List[CanvasNode] = Field(default_factory=list)
    edges: List[CanvasEdge] = Field(default_factory=list)


# ── /suggest ───────────────────────────────────────────────────────────

class ProvenanceSnippet(BaseModel):
    paper_id: str
    page: Optional[int] = None
    figure_id: Optional[str] = None
    source: Optional[str] = None       # "text" | "figure"
    quote: Optional[str] = None


class SuggestedNode(BaseModel):
    """A KG node not yet on the canvas that connects to the query node."""
    kind: Literal["new_node"] = "new_node"
    kg_id: str                          # canonical KG node id
    label: str
    category: Optional[str] = None      # mapped from KG `category`
    suggested_edges: List["SuggestedEdgeFromNew"] = Field(default_factory=list)
    alpha: float                        # 0..1 confidence — frontend renders at this opacity
    rationale: Optional[str] = None
    provenance: List[ProvenanceSnippet] = Field(default_factory=list)


class SuggestedEdgeFromNew(BaseModel):
    """An edge between a newly-proposed KG node and a node already on the canvas."""
    canvas_node_id: str
    direction: Literal["incoming", "outgoing"]
    polarity: Literal["+", "-"]
    weight: int                         # 1..5 from KG
    provenance: List[ProvenanceSnippet] = Field(default_factory=list)


class SuggestedCanvasEdge(BaseModel):
    """A new edge between two nodes already on the user's canvas."""
    kind: Literal["new_edge"] = "new_edge"
    source: str                         # canvas node id
    target: str                         # canvas node id
    polarity: Literal["+", "-"]
    weight: int
    alpha: float
    rationale: Optional[str] = None
    provenance: List[ProvenanceSnippet] = Field(default_factory=list)


class SuggestRequest(BaseModel):
    query_node: CanvasNode              # the node the user clicked / is editing
    canvas: CanvasState                 # current state of their CLD
    max_suggestions: int = 8
    include_new_nodes: bool = True
    include_new_edges: bool = True


class SuggestResponse(BaseModel):
    query_kg_match: Optional[str] = None   # best-matching KG id (if any)
    match_confidence: float = 0.0
    new_nodes: List[SuggestedNode] = Field(default_factory=list)
    new_edges: List[SuggestedCanvasEdge] = Field(default_factory=list)
    latency_ms: int = 0


# ── /causal-query ──────────────────────────────────────────────────────

class CausalQueryRequest(BaseModel):
    source: str                         # canvas node id
    target: Optional[str] = None        # if None, return top-K affected nodes
    direction: Literal["increase", "decrease"] = "increase"
    canvas: CanvasState
    max_paths: int = 4


class CausalPath(BaseModel):
    nodes: List[str]                    # ids in order
    labels: List[str]                   # human labels for narration
    signs: List[Literal["+", "-"]]
    net_sign: Literal["+", "-"]
    strength: float                     # 0..1 (combined damping × weight)
    loop_id: Optional[str] = None


class CausalQueryResponse(BaseModel):
    summary: str                        # LLM-narrated answer
    net_direction: Optional[Literal["increase", "decrease", "ambiguous"]] = None
    paths: List[CausalPath] = Field(default_factory=list)
    loops_involved: List[str] = Field(default_factory=list)


# ── /chat ──────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    canvas: CanvasState


class PendingMutation(BaseModel):
    """A staged change proposed by the chat agent. Frontend renders these
    as a confirm/reject card; on accept it applies the diff to the canvas."""
    id: str
    op: Literal["add_node", "remove_node", "add_edge", "remove_edge"]
    payload: Dict[str, Any]
    summary: str                        # human-readable one-liner


class ChatResponse(BaseModel):
    """Non-streaming variant. The /chat endpoint streams these fields as SSE."""
    reply: str
    mutations: List[PendingMutation] = Field(default_factory=list)


# ── /loop-chat ─────────────────────────────────────────────────────────

class SelectionScope(BaseModel):
    """The Visualise tab's current focus — feeds the loop-chat system
    prompt so the LLM can refuse to discuss anything outside this view.
    """
    # Loop currently highlighted in the Loop badge bar, or "ALL"
    selected_loop_id: Optional[str] = None
    selected_loop_name: Optional[str] = None
    selected_loop_type: Optional[str] = None       # "R" or "B"
    selected_loop_link_pairs: List[List[str]] = Field(default_factory=list)
    # Subsystem filter (substring of `subsystem` field on each node), or "ALL"
    active_subsystem: Optional[str] = None
    # Optional: the user clicked a specific node/edge
    selected_item_kind: Optional[Literal["node", "link"]] = None
    selected_item_label: Optional[str] = None
    # Ids of the nodes currently *visible* on screen (after filters)
    visible_node_ids: List[str] = Field(default_factory=list)


class LoopChatRequest(BaseModel):
    messages: List[ChatMessage]
    canvas: CanvasState
    scope: SelectionScope


# ── /log ───────────────────────────────────────────────────────────────

class LogEventItem(BaseModel):
    """One activity event captured on the client."""
    timestamp: str             # ISO-8601 from the browser clock
    event: str                 # e.g. node_added, chat_user_msg
    payload: Dict[str, Any] = Field(default_factory=dict)


class LogRequest(BaseModel):
    """Batch of events from one session."""
    user_id: str               # filesystem-safe slug of `name`
    name: str                  # raw display name as entered on landing
    session_id: str            # uuid generated by the frontend
    started_at: Optional[str] = None
    events: List[LogEventItem] = Field(default_factory=list)


# Forward refs
SuggestedNode.model_rebuild()
