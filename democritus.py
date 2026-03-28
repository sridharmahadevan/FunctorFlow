from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .core import Diagram
from .macros import (
    DBSquareConfig,
    DemocritusGluingConfig,
    GTNeighborhoodConfig,
    KETBlockConfig,
    db_square,
    democritus_gluing_block,
    gt_neighborhood_block,
    ket_block,
)


class DemocritusLLMClient(Protocol):
    def ask(self, prompt: str) -> str:
        ...

    def ask_batch(self, prompts: Sequence[str]) -> list[str]:
        ...


@dataclass(frozen=True)
class DemocritusLLMConfig:
    base_url: str = "https://api.openai.com"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 256
    temperature: float = 0.7
    max_batch_size: int = 4
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "DemocritusLLMConfig":
        return cls(
            base_url=os.getenv("DEMOC_LLM_BASE_URL", "https://api.openai.com").rstrip("/"),
            model=os.getenv("DEMOC_LLM_MODEL", "gpt-4.1-mini"),
            max_tokens=int(os.getenv("DEMOC_LLM_MAX_TOKENS", "256")),
            temperature=float(os.getenv("DEMOC_LLM_TEMPERATURE", "0.7")),
            max_batch_size=int(os.getenv("DEMOC_LLM_BATCH_SIZE", "4")),
            timeout=int(os.getenv("DEMOC_LLM_TIMEOUT", "120")),
        )


class OpenAICompatibleDemocritusClient:
    """Small standalone OpenAI-compatible chat client for the public Democritus path."""

    def __init__(
        self,
        config: DemocritusLLMConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        cfg = config or DemocritusLLMConfig.from_env()
        self.config = cfg
        self.api_key = api_key or os.getenv(cfg.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{cfg.api_key_env} not set. Export it before running the Democritus pipeline."
            )

    def _single_chat(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise assistant that follows instructions exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        request = urllib.request.Request(
            url=f"{self.config.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network error path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        choices = body.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content") or "").strip()

    def ask(self, prompt: str) -> str:
        return self._single_chat(prompt)

    def ask_batch(self, prompts: Sequence[str]) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), self.config.max_batch_size):
            batch = prompts[start : start + self.config.max_batch_size]
            for prompt in batch:
                outputs.append(self._single_chat(prompt))
        return outputs


@dataclass(frozen=True)
class DemocritusTopicDiscoveryConfig:
    num_root_topics: int = 18
    topics_per_chunk: int = 6
    batch_size: int = 8
    max_chunk_chars: int = 2000
    max_tokens: int = 128


@dataclass(frozen=True)
class DemocritusTopicGraphConfig:
    depth_limit: int = 3
    max_total_topics: int = 100
    batch_size: int = 8
    subtopics_per_topic: int = 10


@dataclass(frozen=True)
class DemocritusQuestionConfig:
    questions_per_topic: int = 2
    batch_size: int = 16
    max_tokens: int = 128


@dataclass(frozen=True)
class DemocritusStatementConfig:
    statements_per_question: int = 2
    batch_size: int = 16
    max_tokens: int = 192


@dataclass(frozen=True)
class DemocritusManifoldConfig:
    gt_depth: int = 2
    normalize_features: bool = True
    random_seed: int = 7
    sample_local_models: int = 3
    local_model_radius: int = 2
    local_model_min_focus_edges: int = 2
    local_model_max_nodes: int = 10
    local_model_max_edges: int = 14


@dataclass(frozen=True)
class DemocritusPipelineConfig:
    domain_name: str = "democritus"
    topic_discovery: DemocritusTopicDiscoveryConfig = field(
        default_factory=DemocritusTopicDiscoveryConfig
    )
    topic_graph: DemocritusTopicGraphConfig = field(default_factory=DemocritusTopicGraphConfig)
    questions: DemocritusQuestionConfig = field(default_factory=DemocritusQuestionConfig)
    statements: DemocritusStatementConfig = field(default_factory=DemocritusStatementConfig)
    manifold: DemocritusManifoldConfig = field(default_factory=DemocritusManifoldConfig)

    @classmethod
    def smoke(cls, *, domain_name: str = "democritus_smoke") -> "DemocritusPipelineConfig":
        return cls(
            domain_name=domain_name,
            topic_discovery=DemocritusTopicDiscoveryConfig(
                num_root_topics=6,
                topics_per_chunk=3,
                batch_size=4,
                max_chunk_chars=1000,
                max_tokens=96,
            ),
            topic_graph=DemocritusTopicGraphConfig(
                depth_limit=1,
                max_total_topics=16,
                batch_size=4,
                subtopics_per_topic=4,
            ),
            questions=DemocritusQuestionConfig(questions_per_topic=2, batch_size=8, max_tokens=96),
            statements=DemocritusStatementConfig(
                statements_per_question=2,
                batch_size=8,
                max_tokens=128,
            ),
            manifold=DemocritusManifoldConfig(gt_depth=1),
        )


@dataclass(frozen=True)
class DemocritusTopicNode:
    topic: str
    parent: str | None
    depth: int


@dataclass(frozen=True)
class DemocritusQuestionRecord:
    topic: str
    path: tuple[str, ...]
    questions: tuple[str, ...]


@dataclass(frozen=True)
class DemocritusStatementRecord:
    topic: str
    path: tuple[str, ...]
    question: str
    statements: tuple[str, ...]


@dataclass(frozen=True)
class DemocritusTriple:
    topic: str
    path: tuple[str, ...]
    question: str
    statement: str
    subj: str
    rel: str
    obj: str
    domain: str


@dataclass(frozen=True)
class DemocritusCausalManifold:
    entities: tuple[str, ...]
    embeddings_2d: np.ndarray
    embeddings_3d: np.ndarray
    feature_matrix: np.ndarray
    relation_names: tuple[str, ...]
    domain_names: tuple[str, ...]
    db_obstruction: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        return {
            "entities": list(self.entities),
            "relation_names": list(self.relation_names),
            "domain_names": list(self.domain_names),
            "db_obstruction": self.db_obstruction,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DemocritusArtifacts:
    config: DemocritusPipelineConfig
    document_text: str
    root_topics: tuple[str, ...]
    topic_graph: tuple[DemocritusTopicNode, ...]
    causal_questions: tuple[DemocritusQuestionRecord, ...]
    causal_statements: tuple[DemocritusStatementRecord, ...]
    relational_triples: tuple[DemocritusTriple, ...]
    manifold: DemocritusCausalManifold
    output_files: dict[str, str] = field(default_factory=dict)


TOPIC_PROMPT = """
You are a scientific editor.

Given the following excerpt from a document, propose {k} short topic phrases
(3-8 words each) that capture its main causal or thematic concerns.

Rules:
- Topics must be phrases, not sentences.
- Do NOT include numbering or bullets.
- Do NOT mention these instructions.
- One topic per line, no extra text.

Excerpt:
\"\"\"{chunk}\"\"\"

Topics:
""".strip()


QUESTION_PREFIXES = ("what", "how", "why", "when", "who", "which")
INSTRUCTION_PHRASES = (
    "use the following format",
    "causal research question",
    "research question:",
    "questions are in the form",
    "the statements should be",
    "focus on the causal relationship",
    "note:",
    "example:",
    "this question",
    "write in the first person",
    "do not repeat",
    "each question must be",
)
REL_PATTERNS = {
    "causes": r"(.+?)\s+causes\s+(.+)",
    "leads_to": r"(.+?)\s+leads to\s+(.+)",
    "increases": r"(.+?)\s+increases\s+(.+)",
    "reduces": r"(.+?)\s+reduces\s+(.+)",
    "affects": r"(.+?)\s+affects\s+(.+)",
    "influences": r"(.+?)\s+influences\s+(.+)",
    "shapes": r"(.+?)\s+shapes\s+(.+)",
    "contributes_to": r"(.+?)\s+contributes to\s+(.+)",
    "correlates_with": r"(.+?)\s+correlates with\s+(.+)",
    "is_associated_with": r"(.+?)\s+is associated with\s+(.+)",
}
CAUSAL_KEYWORDS = (
    "cause",
    "causes",
    "caused",
    "lead to",
    "leads to",
    "led to",
    "increase",
    "increases",
    "increased",
    "reduce",
    "reduces",
    "reduced",
    "affect",
    "affects",
    "affected",
    "influence",
    "influences",
    "influenced",
)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    pdf_path = str(pdf_path)
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    except Exception:
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # pragma: no cover - depends on runtime deps
            raise RuntimeError(f"Could not extract text from PDF {pdf_path!r}") from exc


def chunk_text(text: str, *, max_chars: int = 2000) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if current and length + len(line) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [line]
            length = len(line)
        else:
            current.append(line)
            length += len(line) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [text[:max_chars]]


def _parse_topics(raw: str) -> list[str]:
    topics: list[str] = []
    for line in raw.splitlines():
        candidate = line.strip(" \t*-0123456789.").strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if "topics must be extracted" in lowered or "follow the guidelines strictly" in lowered:
            continue
        word_count = len(candidate.split())
        if 2 <= word_count <= 8:
            topics.append(candidate)
    return topics


def discover_topics_from_text(
    text: str,
    *,
    llm: DemocritusLLMClient,
    config: DemocritusTopicDiscoveryConfig,
) -> list[str]:
    prompts = [
        TOPIC_PROMPT.format(k=config.topics_per_chunk, chunk=chunk)
        for chunk in chunk_text(text, max_chars=config.max_chunk_chars)
    ]
    raw_outputs = llm.ask_batch(prompts)
    counts: Counter[str] = Counter()
    canonical: dict[str, str] = {}
    for raw in raw_outputs:
        for topic in _parse_topics(raw):
            key = topic.lower()
            counts[key] += 1
            canonical.setdefault(key, topic)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [canonical[key] for key, _ in ranked[: config.num_root_topics]]


def discover_topics_from_pdf(
    pdf_path: str | Path,
    *,
    llm: DemocritusLLMClient,
    config: DemocritusTopicDiscoveryConfig,
) -> tuple[str, list[str]]:
    text = extract_text_from_pdf(pdf_path)
    return text, discover_topics_from_text(text, llm=llm, config=config)


def _build_subtopic_prompt(topic: str, count: int) -> str:
    return (
        f'List {count} detailed subtopics related to "{topic}".\n'
        "Write only the subtopics.\n"
        "One per line."
    )


def _parse_subtopics(raw: str) -> list[str]:
    subtopics: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        candidate = line.strip()
        while candidate and candidate[0] in "-•*0123456789. ":
            candidate = candidate[1:].strip()
        if not candidate:
            continue
        if len(candidate.split()) > 8 or len(candidate) < 3:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        subtopics.append(candidate)
    return subtopics


def build_topic_graph(
    root_topics: Sequence[str],
    *,
    llm: DemocritusLLMClient,
    config: DemocritusTopicGraphConfig,
) -> list[DemocritusTopicNode]:
    nodes: dict[str, DemocritusTopicNode] = {
        topic: DemocritusTopicNode(topic=topic, parent=None, depth=0) for topic in root_topics
    }
    queue: deque[str] = deque(root_topics)
    expanded: set[str] = set()

    while queue and len(nodes) < config.max_total_topics:
        parents: list[DemocritusTopicNode] = []
        while queue and len(parents) < config.batch_size:
            parent_topic = queue.popleft()
            parent = nodes[parent_topic]
            if parent.depth >= config.depth_limit or parent.topic in expanded:
                continue
            parents.append(parent)
            expanded.add(parent.topic)
        if not parents:
            break

        prompts = [
            _build_subtopic_prompt(parent.topic, config.subtopics_per_topic) for parent in parents
        ]
        for parent, raw in zip(parents, llm.ask_batch(prompts)):
            for child_topic in _parse_subtopics(raw):
                if len(nodes) >= config.max_total_topics:
                    break
                if child_topic in nodes:
                    continue
                child = DemocritusTopicNode(
                    topic=child_topic,
                    parent=parent.topic,
                    depth=parent.depth + 1,
                )
                nodes[child_topic] = child
                queue.append(child.topic)
    return sorted(nodes.values(), key=lambda node: (node.depth, node.topic.lower()))


def _build_paths(nodes: Sequence[DemocritusTopicNode]) -> dict[str, tuple[str, ...]]:
    parent = {node.topic: node.parent for node in nodes}
    cache: dict[str, tuple[str, ...]] = {}

    def resolve(topic: str) -> tuple[str, ...]:
        if topic in cache:
            return cache[topic]
        parent_topic = parent[topic]
        if parent_topic is None:
            cache[topic] = (topic,)
        else:
            cache[topic] = resolve(parent_topic) + (topic,)
        return cache[topic]

    for topic in parent:
        resolve(topic)
    return cache


def _build_question_prompt(path: Sequence[str], count: int) -> str:
    chain = " -> ".join(path)
    return f"""
Generate {count} distinct causal questions about:

{chain}

Rules:
- One question per line
- Each question must contain a causal verb
  (causes, affects, influences, leads to, reduces, increases)
- No bullets, no numbering
- No explanations, no commentary
- Output exactly {count} lines of questions
""".strip()


def _parse_questions(raw: str, count: int) -> list[str]:
    questions: list[str] = []
    for line in raw.splitlines():
        candidate = line.strip(" -•\t")
        if len(candidate) < 8:
            continue
        if "note:" in candidate.lower():
            continue
        questions.append(candidate)
        if len(questions) >= count:
            break
    return questions


def build_causal_questions(
    topic_graph: Sequence[DemocritusTopicNode],
    *,
    llm: DemocritusLLMClient,
    config: DemocritusQuestionConfig,
) -> list[DemocritusQuestionRecord]:
    paths = _build_paths(topic_graph)
    records: list[DemocritusQuestionRecord] = []
    batch_topics = list(topic_graph)
    for start in range(0, len(batch_topics), config.batch_size):
        batch = batch_topics[start : start + config.batch_size]
        prompts = [
            _build_question_prompt(paths[node.topic], config.questions_per_topic) for node in batch
        ]
        for node, raw in zip(batch, llm.ask_batch(prompts)):
            questions = _parse_questions(raw, config.questions_per_topic)
            if not questions:
                continue
            records.append(
                DemocritusQuestionRecord(
                    topic=node.topic,
                    path=paths[node.topic],
                    questions=tuple(questions),
                )
            )
    return records


def _build_statement_prompt(question: str, count: int) -> str:
    return f"""
You are a causal knowledge generator.

Given the causal research question below, write EXACTLY {count} causal statements.

Each statement must:
- be a declarative sentence,
- describe a cause and an effect,
- contain one of the words: causes, leads to, increases, reduces, affects, influences,
- be scientifically meaningful.

Do NOT:
- repeat or refer to these instructions,
- describe a format or example,
- use bullets or numbering,
- mention anything about questions or statements.

Write exactly {count} sentences, each on its own line.

Question:
"{question}"
""".strip()


def _split_into_sentences(text: str) -> list[str]:
    fragments = re.split(r"[.?!]", text.replace("\n", " "))
    return [fragment.strip() for fragment in fragments if len(fragment.strip()) >= 10]


def _parse_statements(raw: str, count: int) -> list[str]:
    candidates: list[str] = []
    causal_candidates: list[str] = []
    for sentence in _split_into_sentences(raw):
        lowered = sentence.lower()
        if any(bad in lowered for bad in ("format", "note:", "instruction", "example:")):
            continue
        candidate = sentence if sentence.endswith(".") else f"{sentence}."
        candidates.append(candidate)
        if any(keyword in lowered for keyword in CAUSAL_KEYWORDS):
            causal_candidates.append(candidate)
    chosen = causal_candidates or candidates
    return chosen[:count]


def build_causal_statements(
    question_records: Sequence[DemocritusQuestionRecord],
    *,
    llm: DemocritusLLMClient,
    config: DemocritusStatementConfig,
) -> list[DemocritusStatementRecord]:
    flattened: list[tuple[str, tuple[str, ...], str]] = []
    for record in question_records:
        for question in record.questions:
            flattened.append((record.topic, record.path, question))

    outputs: list[DemocritusStatementRecord] = []
    for start in range(0, len(flattened), config.batch_size):
        batch = flattened[start : start + config.batch_size]
        prompts = [
            _build_statement_prompt(question, config.statements_per_question)
            for _, _, question in batch
        ]
        for (topic, path, question), raw in zip(batch, llm.ask_batch(prompts)):
            statements = _parse_statements(raw, config.statements_per_question)
            if not statements:
                continue
            outputs.append(
                DemocritusStatementRecord(
                    topic=topic,
                    path=path,
                    question=question,
                    statements=tuple(statements),
                )
            )
    return outputs


def _clean_text(value: str) -> str:
    return value.strip().strip("\"'“”‘’`").strip()


def extract_triple(statement: str) -> tuple[str, str, str] | None:
    lowered = statement.strip().lower()
    if any(bad in lowered for bad in INSTRUCTION_PHRASES):
        return None
    lowered = lowered.rstrip(".")
    for relation, pattern in REL_PATTERNS.items():
        match = re.search(pattern, lowered)
        if match is None:
            continue
        subj = _clean_text(match.group(1))
        obj = _clean_text(match.group(2))
        if any(subj.startswith(prefix) for prefix in QUESTION_PREFIXES):
            return None
        if len(subj) < 3 or len(obj) < 3:
            return None
        return subj, relation, obj
    return None


def extract_relational_triples(
    records: Sequence[DemocritusStatementRecord],
) -> list[DemocritusTriple]:
    triples: list[DemocritusTriple] = []
    for record in records:
        for statement in record.statements:
            parsed = extract_triple(statement)
            if parsed is None:
                continue
            subj, rel, obj = parsed
            triples.append(
                DemocritusTriple(
                    topic=record.topic,
                    path=record.path,
                    question=record.question,
                    statement=statement,
                    subj=subj,
                    rel=rel,
                    obj=obj,
                    domain=record.path[0],
                )
            )
    return triples


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums


def _project_pca(features: np.ndarray, dims: int) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0, dims), dtype=np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, dims), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[: min(dims, vt.shape[0])].T
    if projection.shape[1] < dims:
        projection = np.pad(projection, ((0, 0), (0, dims - projection.shape[1])))
    return projection.astype(np.float32)


def _compute_db_obstruction(relation_matrices: Sequence[np.ndarray]) -> float:
    if len(relation_matrices) < 2:
        return 0.0
    total = 0.0
    count = 0
    for index, left in enumerate(relation_matrices):
        for right in relation_matrices[index + 1 :]:
            commutator = left @ right - right @ left
            total += float(np.linalg.norm(commutator))
            count += 1
    return total / max(count, 1)


def build_causal_manifold(
    triples: Sequence[DemocritusTriple],
    *,
    config: DemocritusManifoldConfig,
) -> DemocritusCausalManifold:
    entity_names = tuple(sorted({triple.subj for triple in triples} | {triple.obj for triple in triples}))
    relation_names = tuple(sorted({triple.rel for triple in triples}))
    domain_names = tuple(sorted({triple.domain for triple in triples}))
    if not entity_names:
        empty = np.zeros((0, 2), dtype=np.float32)
        return DemocritusCausalManifold(
            entities=(),
            embeddings_2d=empty,
            embeddings_3d=np.zeros((0, 3), dtype=np.float32),
            feature_matrix=np.zeros((0, 0), dtype=np.float32),
            relation_names=(),
            domain_names=(),
            db_obstruction=0.0,
            metadata={"gt_depth": config.gt_depth, "num_triples": 0},
        )

    entity_index = {name: idx for idx, name in enumerate(entity_names)}
    relation_index = {name: idx for idx, name in enumerate(relation_names)}
    domain_index = {name: idx for idx, name in enumerate(domain_names)}

    num_entities = len(entity_names)
    adjacency = np.zeros((num_entities, num_entities), dtype=np.float32)
    outgoing_relation = np.zeros((num_entities, len(relation_names)), dtype=np.float32)
    incoming_relation = np.zeros((num_entities, len(relation_names)), dtype=np.float32)
    domain_features = np.zeros((num_entities, len(domain_names)), dtype=np.float32)
    relation_matrices = [np.zeros((num_entities, num_entities), dtype=np.float32) for _ in relation_names]

    for triple in triples:
        source = entity_index[triple.subj]
        target = entity_index[triple.obj]
        relation_id = relation_index[triple.rel]
        domain_id = domain_index[triple.domain]
        adjacency[source, target] += 1.0
        adjacency[target, source] += 1.0
        relation_matrices[relation_id][source, target] += 1.0
        outgoing_relation[source, relation_id] += 1.0
        incoming_relation[target, relation_id] += 1.0
        domain_features[source, domain_id] += 1.0
        domain_features[target, domain_id] += 1.0

    degree = adjacency.sum(axis=1, keepdims=True)
    base_features = np.concatenate(
        [degree, outgoing_relation, incoming_relation, domain_features],
        axis=1,
    )
    if config.normalize_features and base_features.size:
        norms = np.linalg.norm(base_features, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        base_features = base_features / norms

    diffusion = _row_normalize(adjacency + np.eye(num_entities, dtype=np.float32))
    ket_features = diffusion @ base_features
    refined = ket_features
    for _ in range(max(config.gt_depth, 1) - 1):
        refined = diffusion @ refined
    refined = 0.5 * (refined + ket_features)

    return DemocritusCausalManifold(
        entities=entity_names,
        embeddings_2d=_project_pca(refined, 2),
        embeddings_3d=_project_pca(refined, 3),
        feature_matrix=refined.astype(np.float32),
        relation_names=relation_names,
        domain_names=domain_names,
        db_obstruction=_compute_db_obstruction(relation_matrices),
        metadata={
            "gt_depth": config.gt_depth,
            "num_triples": len(triples),
            "num_entities": len(entity_names),
            "num_relations": len(relation_names),
            "num_domains": len(domain_names),
        },
    )


def build_democritus_pipeline_diagram(name: str = "DemocritusPipeline") -> Diagram:
    diagram = Diagram(name)
    diagram.object("DocumentText", kind="document_text")
    diagram.object("TopicGraph", kind="topic_graph")
    diagram.object("CausalQuestions", kind="questions")
    diagram.object("CausalStatements", kind="statements")

    topic_sections = diagram.include(
        ket_block(
            KETBlockConfig(
                name="DemocritusTopicAggregation",
                source_object="RootTopicVotes",
                relation_object="TopicChunkIncidence",
                target_object="LocalTopicSections",
                aggregate_name="aggregate_topics",
                reducer="concat",
            )
        ),
        namespace="topic_sections",
    )
    diagram.morphism(
        "discover_root_topics",
        "DocumentText",
        topic_sections.port_spec("input"),
        description="LLM-backed topic discovery over PDF text chunks.",
    )

    diagram.morphism(
        "expand_topic_graph",
        topic_sections.port_spec("output"),
        "TopicGraph",
        description="Expand root topics into a topic graph via breadth-first prompting.",
    )
    diagram.morphism(
        "generate_causal_questions",
        "TopicGraph",
        "CausalQuestions",
        description="Generate causal questions for each topic path.",
    )
    diagram.morphism(
        "generate_causal_statements",
        "CausalQuestions",
        "CausalStatements",
        description="Convert causal questions into declarative causal statements.",
    )
    refined_claims = diagram.include(
        gt_neighborhood_block(
            GTNeighborhoodConfig(
                name="DemocritusGTRefinement",
                token_object="RelationalTriples",
                relation_object="ClaimNeighborhood",
                message_object="ClaimMessages",
                target_object="RefinedClaims",
                aggregate_name="refine_claims",
            )
        ),
        namespace="gt_refine",
    )
    diagram.morphism(
        "extract_relational_triples",
        "CausalStatements",
        refined_claims.port_spec("input"),
        description="Parse causal statements into (subject, relation, object) triples.",
    )
    diagram.register_adapter(
        "claims_to_local_sections",
        source_type="token_state",
        target_type="local_sections",
        implementation=lambda value: value,
        description="Treat GT-refined claim states as local sheaf sections for gluing.",
    )

    glued = diagram.include(
        democritus_gluing_block(
            DemocritusGluingConfig(
                name="DemocritusGluing",
                source_object="RefinedClaims",
                relation_object="OverlapRegions",
                target_object="GlobalManifold",
                gluing_name="glue_claims",
                reducer="set_union",
            )
        ),
        namespace="glue",
        object_aliases={
            "RefinedClaims": refined_claims.port_spec("output"),
        },
    )
    diagram.register_adapter(
        "global_manifold_to_state",
        source_type="global_state",
        target_type="state",
        implementation=lambda value: value,
        description="View the glued global manifold as a DB state for consistency checks.",
    )

    consistency = diagram.include(
        db_square(
            DBSquareConfig(
                name="DemocritusConsistency",
                state_object="GlobalManifold",
                first_morphism="project_domains",
                second_morphism="project_relations",
                left_path="domain_then_relation",
                right_path="relation_then_domain",
                loss_name="gluing_obstruction",
            )
        ),
        namespace="consistency",
        object_aliases={"GlobalManifold": glued.port_spec("output")},
    )

    diagram.expose_port("document", "DocumentText", direction="input", port_type="document_text")
    diagram.expose_port(
        "topic_votes",
        "discover_root_topics",
        direction="internal",
        port_type="local_sections",
    )
    diagram.expose_port("triples", "extract_relational_triples", direction="output", port_type="token_state")
    diagram.expose_port("manifold", glued.port_spec("output"), direction="output", port_type="global_state")
    diagram.expose_port(
        "obstruction",
        consistency.loss("gluing_obstruction"),
        direction="output",
        kind="loss",
        port_type="loss",
    )
    return diagram


def _default_llm_client() -> OpenAICompatibleDemocritusClient:
    return OpenAICompatibleDemocritusClient(DemocritusLLMConfig.from_env())


def _run_democritus_pipeline(
    text: str,
    *,
    root_topics: Sequence[str] | None,
    llm: DemocritusLLMClient,
    config: DemocritusPipelineConfig,
    outdir: str | Path | None,
) -> DemocritusArtifacts:
    topics = list(root_topics) if root_topics is not None else discover_topics_from_text(
        text,
        llm=llm,
        config=config.topic_discovery,
    )
    topic_graph = build_topic_graph(
        topics,
        llm=llm,
        config=config.topic_graph,
    )
    causal_questions = build_causal_questions(
        topic_graph,
        llm=llm,
        config=config.questions,
    )
    causal_statements = build_causal_statements(
        causal_questions,
        llm=llm,
        config=config.statements,
    )
    relational_triples = extract_relational_triples(causal_statements)
    manifold = build_causal_manifold(
        relational_triples,
        config=config.manifold,
    )

    artifacts = DemocritusArtifacts(
        config=config,
        document_text=text,
        root_topics=tuple(topics),
        topic_graph=tuple(topic_graph),
        causal_questions=tuple(causal_questions),
        causal_statements=tuple(causal_statements),
        relational_triples=tuple(relational_triples),
        manifold=manifold,
        output_files={},
    )
    if outdir is not None:
        return write_democritus_artifacts(artifacts, outdir)
    return artifacts


def run_democritus_pipeline_from_text(
    text: str,
    *,
    llm: DemocritusLLMClient | None = None,
    config: DemocritusPipelineConfig | None = None,
    outdir: str | Path | None = None,
) -> DemocritusArtifacts:
    pipeline_config = config or DemocritusPipelineConfig()
    llm_client = llm or _default_llm_client()
    return _run_democritus_pipeline(
        text,
        root_topics=None,
        llm=llm_client,
        config=pipeline_config,
        outdir=outdir,
    )


def run_democritus_pipeline_from_pdf(
    pdf_path: str | Path,
    *,
    llm: DemocritusLLMClient | None = None,
    config: DemocritusPipelineConfig | None = None,
    outdir: str | Path | None = None,
) -> DemocritusArtifacts:
    pipeline_config = config or DemocritusPipelineConfig()
    llm_client = llm or _default_llm_client()
    text, root_topics = discover_topics_from_pdf(
        pdf_path,
        llm=llm_client,
        config=pipeline_config.topic_discovery,
    )
    return _run_democritus_pipeline(
        text,
        root_topics=root_topics,
        llm=llm_client,
        config=pipeline_config,
        outdir=outdir,
    )


def run_democritus_pipelines_from_pdf_directory(
    pdf_dir: str | Path,
    *,
    llm: DemocritusLLMClient | None = None,
    config: DemocritusPipelineConfig | None = None,
    outdir: str | Path | None = None,
) -> dict[str, DemocritusArtifacts]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)
    if not pdf_dir.is_dir():
        raise NotADirectoryError(pdf_dir)

    pdf_paths = sorted(path for path in pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in directory: {pdf_dir}")

    llm_client = llm or _default_llm_client()
    base_config = config or DemocritusPipelineConfig(domain_name=pdf_dir.name)
    output_root = Path(outdir) if outdir is not None else Path("FunctorFlow/democritus_runs")
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, DemocritusArtifacts] = {}
    for index, pdf_path in enumerate(pdf_paths, start=1):
        run_slug = _slugify_plot_name(pdf_path.stem, max_chars=80)
        run_name = f"{index:02d}_{run_slug}"
        run_config = replace(base_config, domain_name=f"{base_config.domain_name}_{run_slug}")
        results[str(pdf_path)] = run_democritus_pipeline_from_pdf(
            pdf_path,
            llm=llm_client,
            config=run_config,
            outdir=output_root / run_name,
        )
    return results


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _shorten_plot_label(label: str, *, max_chars: int = 52) -> str:
    collapsed = re.sub(r"\s+", " ", label).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    if max_chars <= 3:
        return collapsed[:max_chars]
    return collapsed[: max_chars - 3].rstrip() + "..."


def _select_manifold_label_indices(
    points: np.ndarray,
    *,
    max_labels: int = 18,
    grid_shape: tuple[int, int] = (6, 4),
    min_separation: float = 0.18,
) -> list[int]:
    num_points = int(points.shape[0])
    if num_points <= max_labels:
        return list(range(num_points))
    if num_points == 0:
        return []

    minima = points.min(axis=0)
    spans = points.max(axis=0) - minima
    spans[spans == 0.0] = 1.0
    normalized = (points - minima) / spans
    centroid = normalized.mean(axis=0)
    distances = np.linalg.norm(normalized - centroid, axis=1)

    grid_x, grid_y = grid_shape
    candidate_map: dict[tuple[int, int], int] = {}
    for index, point in enumerate(normalized):
        cell_x = min(int(point[0] * grid_x), grid_x - 1)
        cell_y = min(int(point[1] * grid_y), grid_y - 1)
        cell = (cell_x, cell_y)
        current = candidate_map.get(cell)
        if current is None or distances[index] > distances[current]:
            candidate_map[cell] = index

    selected: list[int] = []
    for index in sorted(candidate_map.values(), key=lambda item: float(distances[item]), reverse=True):
        if len(selected) >= max_labels:
            break
        if all(np.linalg.norm(normalized[index] - normalized[chosen]) >= min_separation for chosen in selected):
            selected.append(index)

    if len(selected) < max_labels:
        for index in np.argsort(-distances):
            candidate = int(index)
            if candidate in selected:
                continue
            if all(
                np.linalg.norm(normalized[candidate] - normalized[chosen]) >= min_separation * 0.65
                for chosen in selected
            ):
                selected.append(candidate)
            if len(selected) >= max_labels:
                break

    if len(selected) < max_labels:
        for index in np.argsort(-distances):
            candidate = int(index)
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= max_labels:
                break

    return sorted(selected, key=lambda item: (float(points[item, 0]), float(points[item, 1])))


def _expand_axis_limits(values: np.ndarray, *, padding_ratio: float = 0.12) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    if span == 0.0:
        span = 1.0
    padding = span * padding_ratio
    return lower - padding, upper + padding


def _slugify_plot_name(value: str, *, max_chars: int = 56) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        slug = "local_model"
    return slug[:max_chars].rstrip("_") or "local_model"


def _build_local_causal_graph(triples: Sequence[DemocritusTriple]) -> tuple[Any, set[str]]:
    try:
        import networkx as nx
    except ModuleNotFoundError:  # pragma: no cover - depends on runtime deps
        return None, set()

    graph = nx.DiGraph()
    topic_nodes: set[str] = set()
    for triple in triples:
        graph.add_node(triple.subj)
        graph.add_node(triple.obj)
        graph.add_edge(triple.subj, triple.obj, rel=triple.rel)

        for topic_node in (triple.topic, *triple.path):
            if not topic_node:
                continue
            topic_nodes.add(topic_node)
            graph.add_node(topic_node)
            graph.add_edge(topic_node, triple.subj, rel="has_subj")
            graph.add_edge(topic_node, triple.obj, rel="has_obj")

    return graph, topic_nodes


def _select_local_model_focuses_from_graph(
    graph: Any,
    *,
    topic_nodes: set[str],
    max_models: int,
    min_focus_edges: int,
) -> list[str]:
    if graph is None or max_models <= 0:
        return []

    ranked = sorted(
        graph.degree,
        key=lambda item: (
            1 if item[0] in topic_nodes else 0,
            item[1],
            -len(str(item[0])),
            str(item[0]).lower(),
        ),
        reverse=True,
    )
    selected: list[str] = []
    for node, degree in ranked:
        if degree < min_focus_edges:
            continue
        if node in selected:
            continue
        selected.append(str(node))
        if len(selected) >= max_models:
            break
    return selected


def _extract_local_model_graph(
    graph: Any,
    *,
    focus: str,
    radius: int,
    max_nodes: int,
) -> Any:
    try:
        import networkx as nx
    except ModuleNotFoundError:  # pragma: no cover - depends on runtime deps
        return None

    if graph is None or focus not in graph:
        return None

    local_graph = nx.ego_graph(graph, focus, radius=max(radius, 1), undirected=False)
    if local_graph.number_of_nodes() > 0:
        undirected = local_graph.to_undirected()
        component = next(component for component in nx.connected_components(undirected) if focus in component)
        local_graph = local_graph.subgraph(component).copy()

    if local_graph.number_of_nodes() > max_nodes:
        ranked = sorted(local_graph.degree, key=lambda item: item[1], reverse=True)
        keep = {focus} | {node for node, _ in ranked[: max_nodes - 1]}
        local_graph = local_graph.subgraph(keep).copy()
    return local_graph


def _select_local_model_focuses(
    triples: Sequence[DemocritusTriple],
    *,
    max_models: int,
    min_focus_edges: int,
) -> list[str]:
    if max_models <= 0:
        return []

    incident_counts: Counter[str] = Counter()
    neighbor_sets: dict[str, set[str]] = {}
    incoming_counts: Counter[str] = Counter()
    outgoing_counts: Counter[str] = Counter()
    for triple in triples:
        incident_counts[triple.subj] += 1
        incident_counts[triple.obj] += 1
        outgoing_counts[triple.subj] += 1
        incoming_counts[triple.obj] += 1
        neighbor_sets.setdefault(triple.subj, set()).add(triple.obj)
        neighbor_sets.setdefault(triple.obj, set()).add(triple.subj)

    ranked = sorted(
        incident_counts,
        key=lambda entity: (
            incident_counts[entity],
            len(neighbor_sets.get(entity, set())),
            outgoing_counts[entity] + incoming_counts[entity],
            entity,
        ),
        reverse=True,
    )

    selected: list[str] = []
    selected_neighborhoods: list[set[str]] = []
    for entity in ranked:
        neighborhood = set(neighbor_sets.get(entity, set()))
        neighborhood.add(entity)
        if incident_counts[entity] < min_focus_edges:
            continue
        if any(
            len(neighborhood & existing) / max(1, min(len(neighborhood), len(existing))) >= 0.8
            for existing in selected_neighborhoods
        ):
            continue
        selected.append(entity)
        selected_neighborhoods.append(neighborhood)
        if len(selected) >= max_models:
            break

    return selected


def _build_local_model_spec(
    triples: Sequence[DemocritusTriple],
    *,
    focus: str,
    radius: int,
    max_nodes: int,
    max_edges: int,
) -> dict[str, Any] | None:
    adjacency: dict[str, set[str]] = {}
    incident_counts: Counter[str] = Counter()
    for triple in triples:
        adjacency.setdefault(triple.subj, set()).add(triple.obj)
        adjacency.setdefault(triple.obj, set()).add(triple.subj)
        incident_counts[triple.subj] += 1
        incident_counts[triple.obj] += 1

    if focus not in adjacency:
        return None

    max_radius = max(radius, 1)
    visited = {focus}
    frontier = {focus}
    distances = {focus: 0}
    for depth in range(1, max_radius + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in adjacency.get(node, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.add(neighbor)
                distances[neighbor] = depth
        if not next_frontier:
            break
        frontier = next_frontier

    keep_nodes = {focus}
    ranked_neighbors = sorted(
        (node for node in visited if node != focus),
        key=lambda node: (
            distances.get(node, max_radius + 1),
            -incident_counts[node],
            node,
        ),
    )
    for neighbor in ranked_neighbors[: max(max_nodes - 1, 0)]:
        keep_nodes.add(neighbor)

    edge_rows: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    candidate_triples = [
        triple
        for triple in triples
        if triple.subj in keep_nodes and triple.obj in keep_nodes
    ]
    candidate_triples.sort(
        key=lambda triple: (
            0 if triple.subj == focus or triple.obj == focus else 1,
            min(distances.get(triple.subj, max_radius + 1), distances.get(triple.obj, max_radius + 1)),
            triple.subj != focus,
            triple.obj != focus,
            triple.rel,
            triple.subj,
            triple.obj,
        )
    )

    for triple in candidate_triples:
        edge_key = (triple.subj, triple.obj, triple.rel)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_rows.append(
            {
                "src": triple.subj,
                "dst": triple.obj,
                "rel": triple.rel,
                "topic": triple.topic,
                "statement": triple.statement,
                "domain": triple.domain,
            }
        )
        if len(edge_rows) >= max_edges:
            break

    if not edge_rows:
        return None

    active_nodes = {focus}
    for edge in edge_rows:
        active_nodes.add(edge["src"])
        active_nodes.add(edge["dst"])

    return {
        "focus": focus,
        "nodes": sorted(active_nodes),
        "edges": edge_rows,
        "num_nodes": len(active_nodes),
        "num_edges": len(edge_rows),
    }


def _stacked_y_positions(count: int, *, spacing: float = 1.1) -> list[float]:
    if count <= 0:
        return []
    center = (count - 1) / 2.0
    return [spacing * (center - index) for index in range(count)]


def _compute_local_model_layout(model: dict[str, Any]) -> dict[str, tuple[float, float]]:
    focus = str(model["focus"])
    nodes = [str(node) for node in model["nodes"] if str(node) != focus]
    incoming: set[str] = set()
    outgoing: set[str] = set()
    for edge in model["edges"]:
        src = str(edge["src"])
        dst = str(edge["dst"])
        if dst == focus and src != focus:
            incoming.add(src)
        if src == focus and dst != focus:
            outgoing.add(dst)

    both = incoming & outgoing
    left = sorted(incoming - both)
    right = sorted(outgoing - both)
    center = sorted(both)
    remaining = sorted(set(nodes) - set(left) - set(right) - set(center))
    if remaining:
        center.extend(remaining)

    positions: dict[str, tuple[float, float]] = {focus: (0.0, 0.0)}
    for name, y in zip(left, _stacked_y_positions(len(left))):
        positions[name] = (-2.2, y)
    for name, y in zip(right, _stacked_y_positions(len(right))):
        positions[name] = (2.2, y)

    center_top = center[::2]
    center_bottom = center[1::2]
    for name, y in zip(center_top, _stacked_y_positions(len(center_top), spacing=1.0)):
        positions[name] = (0.0, y + 1.55)
    for name, y in zip(center_bottom, _stacked_y_positions(len(center_bottom), spacing=1.0)):
        positions[name] = (0.0, y - 1.55)
    return positions


def _render_local_model_png(model: dict[str, Any], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover - depends on runtime deps
        return False

    positions = _compute_local_model_layout(model)
    if not positions:
        return False

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.set_facecolor("white")
    axis.axis("off")

    for edge_index, edge in enumerate(model["edges"]):
        src = str(edge["src"])
        dst = str(edge["dst"])
        if src not in positions or dst not in positions:
            continue
        start = positions[src]
        end = positions[dst]
        reverse_exists = any(
            other["src"] == dst and other["dst"] == src for other in model["edges"]
        )
        curvature = 0.18 if reverse_exists and src > dst else (-0.18 if reverse_exists else 0.0)
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#d1495b",
                "linewidth": 1.7,
                "alpha": 0.9,
                "shrinkA": 20,
                "shrinkB": 20,
                "connectionstyle": f"arc3,rad={curvature}",
            },
            zorder=1,
        )

        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        normal_x = end[1] - start[1]
        normal_y = start[0] - end[0]
        normal_norm = (normal_x ** 2 + normal_y ** 2) ** 0.5 or 1.0
        scale = 0.12 + 0.02 * (edge_index % 2)
        mid_x += scale * normal_x / normal_norm
        mid_y += scale * normal_y / normal_norm
        axis.text(
            mid_x,
            mid_y,
            _shorten_plot_label(str(edge["rel"]).replace("_", " "), max_chars=18),
            ha="center",
            va="center",
            fontsize=8,
            color="#6b2737",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=3,
        )

    xs = np.array([point[0] for point in positions.values()], dtype=np.float32)
    ys = np.array([point[1] for point in positions.values()], dtype=np.float32)
    for node, (x_pos, y_pos) in positions.items():
        is_focus = node == model["focus"]
        axis.scatter(
            [x_pos],
            [y_pos],
            s=900 if is_focus else 560,
            color="#f4c95d" if is_focus else "#ffe08a",
            edgecolors="#cc3f0c" if is_focus else "#d1495b",
            linewidths=2.0 if is_focus else 1.4,
            zorder=2,
        )
        axis.text(
            x_pos,
            y_pos,
            _shorten_plot_label(node, max_chars=28),
            ha="center",
            va="center",
            fontsize=9 if is_focus else 8,
            color="#153243",
            zorder=4,
        )

    axis.set_xlim(*_expand_axis_limits(xs, padding_ratio=0.22))
    axis.set_ylim(*_expand_axis_limits(ys, padding_ratio=0.25))
    axis.set_title(
        f"Democritus local causal model\nFocus: {_shorten_plot_label(str(model['focus']), max_chars=56)}",
        fontsize=12,
        color="#153243",
    )
    axis.text(
        0.99,
        0.02,
        f"{model['num_nodes']} nodes | {model['num_edges']} edges",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#52606d",
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return True


def _render_local_model_graph_png(local_graph: Any, *, focus: str, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ModuleNotFoundError:  # pragma: no cover - depends on runtime deps
        return False

    if local_graph is None or local_graph.number_of_nodes() == 0:
        return False

    positions = nx.spring_layout(local_graph, seed=0, k=0.8)

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.set_facecolor("white")
    axis.axis("off")

    for source, target in local_graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        axis.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "red",
                "linewidth": 1.5,
                "shrinkA": 5,
                "shrinkB": 5,
            },
        )

    node_sizes = [700 if node == focus else 400 for node in local_graph.nodes()]
    nx.draw_networkx_nodes(
        local_graph,
        positions,
        node_color="gold",
        node_size=node_sizes,
        edgecolors="red",
        linewidths=1.5,
        ax=axis,
    )
    nx.draw_networkx_labels(
        local_graph,
        positions,
        labels={node: _shorten_plot_label(str(node), max_chars=42) for node in local_graph.nodes()},
        font_size=9,
        font_color="blue",
        ax=axis,
    )

    axis.set_title(
        f"Democritus: Causality from Large Language Models\nLocal causal neighborhood of: {_shorten_plot_label(focus, max_chars=60)}",
        color="blue",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=300)
    plt.close(figure)
    return True


def _render_sample_local_models(
    triples: Sequence[DemocritusTriple],
    *,
    outdir: Path,
    max_models: int,
    radius: int,
    min_focus_edges: int,
    max_nodes: int,
    max_edges: int,
) -> list[dict[str, Any]]:
    if not triples or max_models <= 0:
        return []

    graph, topic_nodes = _build_local_causal_graph(triples)
    if graph is None:
        return []

    models_dir = outdir / "local_causal_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for existing in models_dir.iterdir():
        if existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)
    rendered: list[dict[str, Any]] = []
    for index, focus in enumerate(
        _select_local_model_focuses_from_graph(
            graph,
            topic_nodes=topic_nodes,
            max_models=max_models,
            min_focus_edges=min_focus_edges,
        ),
        start=1,
    ):
        local_graph = _extract_local_model_graph(
            graph,
            focus=focus,
            radius=radius,
            max_nodes=max_nodes,
        )
        if local_graph is None or local_graph.number_of_edges() == 0:
            continue
        png_path = models_dir / f"local_causal_model_{index:02d}_{_slugify_plot_name(focus)}.png"
        if not _render_local_model_graph_png(local_graph, focus=focus, out_path=png_path):
            return []
        edge_count = 0
        for source, target, data in local_graph.edges(data=True):
            if edge_count >= max_edges:
                break
            edge_count += 1
        rendered.append(
            {
                "focus": focus,
                "png": str(png_path),
                "num_nodes": local_graph.number_of_nodes(),
                "num_edges": local_graph.number_of_edges(),
                "nodes": [str(node) for node in local_graph.nodes()],
            }
        )
    return rendered


def _render_manifold_png(manifold: DemocritusCausalManifold, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover - depends on runtime deps
        return False

    if manifold.embeddings_2d.shape[0] == 0:
        return False

    points = manifold.embeddings_2d
    label_indices = _select_manifold_label_indices(points)

    figure, axis = plt.subplots(figsize=(11, 8))
    axis.scatter(points[:, 0], points[:, 1], s=26, color="#4c78a8", alpha=0.55, edgecolors="none")
    if label_indices:
        highlighted = points[label_indices]
        axis.scatter(
            highlighted[:, 0],
            highlighted[:, 1],
            s=48,
            color="#e45756",
            alpha=0.95,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    center = points.mean(axis=0)
    for index in label_indices:
        point = points[index]
        offset_x = 12 if point[0] >= center[0] else -12
        offset_y = 10 if point[1] >= center[1] else -10
        axis.annotate(
            _shorten_plot_label(manifold.entities[index]),
            tuple(point),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            alpha=0.95,
            ha="left" if offset_x > 0 else "right",
            va="bottom" if offset_y > 0 else "top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops={"arrowstyle": "-", "color": "#9aa5b1", "linewidth": 0.7, "alpha": 0.8},
            zorder=4,
        )

    axis.set_title("Democritus causal manifold")
    axis.set_xlabel("Axis 1")
    axis.set_ylabel("Axis 2")
    axis.set_xlim(*_expand_axis_limits(points[:, 0]))
    axis.set_ylim(*_expand_axis_limits(points[:, 1]))
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.text(
        0.99,
        0.01,
        f"Annotated {len(label_indices)} of {len(manifold.entities)} entities",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#52606d",
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return True


def write_democritus_artifacts(
    artifacts: DemocritusArtifacts,
    outdir: str | Path,
) -> DemocritusArtifacts:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    root_topics_path = outdir / "root_topics.txt"
    root_topics_path.write_text("\n".join(artifacts.root_topics) + "\n", encoding="utf-8")

    topic_graph_path = outdir / "topic_graph.jsonl"
    _write_jsonl(topic_graph_path, [asdict(node) for node in artifacts.topic_graph])

    questions_path = outdir / "causal_questions.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "topic": record.topic,
                "path": list(record.path),
                "questions": list(record.questions),
            }
            for record in artifacts.causal_questions
        ],
    )

    statements_path = outdir / "causal_statements.jsonl"
    _write_jsonl(
        statements_path,
        [
            {
                "topic": record.topic,
                "path": list(record.path),
                "question": record.question,
                "statements": list(record.statements),
            }
            for record in artifacts.causal_statements
        ],
    )

    triples_path = outdir / "relational_triples.jsonl"
    _write_jsonl(
        triples_path,
        [
            {
                "topic": triple.topic,
                "path": list(triple.path),
                "question": triple.question,
                "statement": triple.statement,
                "subj": triple.subj,
                "rel": triple.rel,
                "obj": triple.obj,
                "domain": triple.domain,
            }
            for triple in artifacts.relational_triples
        ],
    )

    manifold_path = outdir / "manifold.npz"
    np.savez(
        manifold_path,
        entities=np.array(artifacts.manifold.entities, dtype=object),
        embeddings_2d=artifacts.manifold.embeddings_2d,
        embeddings_3d=artifacts.manifold.embeddings_3d,
        feature_matrix=artifacts.manifold.feature_matrix,
        relation_names=np.array(artifacts.manifold.relation_names, dtype=object),
        domain_names=np.array(artifacts.manifold.domain_names, dtype=object),
    )

    summary_path = outdir / "manifold_summary.json"
    summary_path.write_text(
        json.dumps(artifacts.manifold.to_summary(), indent=2),
        encoding="utf-8",
    )

    diagram_path = outdir / "democritus_diagram.json"
    diagram_path.write_text(
        json.dumps(build_democritus_pipeline_diagram().to_ir().as_dict(), indent=2),
        encoding="utf-8",
    )

    output_files = {
        "root_topics": str(root_topics_path),
        "topic_graph": str(topic_graph_path),
        "causal_questions": str(questions_path),
        "causal_statements": str(statements_path),
        "relational_triples": str(triples_path),
        "manifold": str(manifold_path),
        "manifold_summary": str(summary_path),
        "diagram": str(diagram_path),
    }

    plot_path = outdir / "causal_manifold.png"
    if _render_manifold_png(artifacts.manifold, plot_path):
        output_files["plot"] = str(plot_path)

    local_model_entries = _render_sample_local_models(
        artifacts.relational_triples,
        outdir=outdir,
        max_models=artifacts.config.manifold.sample_local_models,
        radius=artifacts.config.manifold.local_model_radius,
        min_focus_edges=artifacts.config.manifold.local_model_min_focus_edges,
        max_nodes=artifacts.config.manifold.local_model_max_nodes,
        max_edges=artifacts.config.manifold.local_model_max_edges,
    )
    if local_model_entries:
        local_models_manifest_path = outdir / "local_causal_models.json"
        local_models_manifest_path.write_text(
            json.dumps(local_model_entries, indent=2),
            encoding="utf-8",
        )
        output_files["local_causal_models_manifest"] = str(local_models_manifest_path)
        for index, entry in enumerate(local_model_entries, start=1):
            output_files[f"local_causal_model_{index:02d}"] = str(entry["png"])

    return DemocritusArtifacts(
        config=artifacts.config,
        document_text=artifacts.document_text,
        root_topics=artifacts.root_topics,
        topic_graph=artifacts.topic_graph,
        causal_questions=artifacts.causal_questions,
        causal_statements=artifacts.causal_statements,
        relational_triples=artifacts.relational_triples,
        manifold=artifacts.manifold,
        output_files=output_files,
    )
