"""Backend and layout helpers for the IQM Garnet port."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Optional

import numpy as np

from .constants import DEFAULT_BACKEND_ALIAS, DEFAULT_BACKEND_URL


def import_qiskit_transpile() -> Any:
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise ImportError("Qiskit is required to transpile Garnet circuits.") from exc
    return transpile


def import_iqm() -> tuple[Any, Any]:
    try:
        from iqm.qiskit_iqm import IQMFakeGarnet, IQMProvider
    except ImportError as exc:
        raise ImportError("iqm-client[qiskit] is required for IQM Garnet integration.") from exc
    return IQMFakeGarnet, IQMProvider


@dataclass(frozen=True)
class LayoutSelection:
    logical_to_physical: tuple[int, ...]
    ring_supported: bool
    degraded: bool
    cz_error_score: float
    coherence_score: float
    reason: str


def _extract_coupling_edges(backend: Any) -> list[tuple[int, int]]:
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None:
        edges = coupling_map.get_edges()
        if edges:
            return [(int(a), int(b)) for a, b in edges]
    target = getattr(backend, "target", None)
    if target is not None:
        build_map = getattr(target, "build_coupling_map", None)
        if build_map is not None:
            generated = build_map()
            if generated is not None:
                return [(int(a), int(b)) for a, b in generated.get_edges()]
    raise ValueError("Backend does not expose a coupling map.")


def _undirected_graph(edges: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        graph[int(source)].add(int(target))
        graph[int(target)].add(int(source))
    return graph


def _gate_error(backend: Any, gate_name: str, qubits: tuple[int, int]) -> float:
    props = getattr(backend, "properties", None)
    if callable(props):
        properties = props()
        if properties is not None:
            gate_error = getattr(properties, "gate_error", None)
            if callable(gate_error):
                try:
                    value = gate_error(gate_name, list(qubits))
                    if value is not None:
                        return float(value)
                except Exception:
                    pass
    return 1.0


def _qubit_coherence_score(backend: Any, qubit: int) -> float:
    props = getattr(backend, "properties", None)
    if callable(props):
        properties = props()
        if properties is not None:
            t1_getter = getattr(properties, "t1", None)
            t2_getter = getattr(properties, "t2", None)
            try:
                t1 = float(t1_getter(qubit)) if callable(t1_getter) else 0.0
                t2 = float(t2_getter(qubit)) if callable(t2_getter) else 0.0
                return t1 + t2
            except Exception:
                return 0.0
    return 0.0


def _cycle_score(backend: Any, cycle: list[int]) -> tuple[float, float]:
    error = 0.0
    coherence = 0.0
    for index, qubit in enumerate(cycle):
        next_qubit = cycle[(index + 1) % len(cycle)]
        error += _gate_error(backend, "cz", (qubit, next_qubit))
        coherence += _qubit_coherence_score(backend, qubit)
    return float(error), float(coherence)


def _canonical_cycle(cycle: list[int]) -> tuple[int, ...]:
    variants = []
    for candidate in (cycle, list(reversed(cycle))):
        best = min(candidate[index:] + candidate[:index] for index in range(len(candidate)))
        variants.append(tuple(best))
    return min(variants)


def _find_simple_cycles(graph: dict[int, set[int]], size: int) -> list[list[int]]:
    cycles: set[tuple[int, ...]] = set()

    def dfs(start: int, current: int, path: list[int], used: set[int]) -> None:
        if len(path) == size:
            if start in graph[current]:
                cycles.add(_canonical_cycle(path[:]))
            return
        for neighbor in sorted(graph[current]):
            if neighbor < start:
                continue
            if neighbor in used:
                continue
            used.add(neighbor)
            path.append(neighbor)
            dfs(start, neighbor, path, used)
            path.pop()
            used.remove(neighbor)

    for start in sorted(graph):
        dfs(start, start, [start], {start})
    return [list(cycle) for cycle in sorted(cycles)]


def _connected_subset_order(graph: dict[int, set[int]], subset: Iterable[int]) -> Optional[list[int]]:
    subset_nodes = set(int(node) for node in subset)
    if not subset_nodes:
        return None
    start = min(subset_nodes)
    queue: deque[int] = deque([start])
    visited = {start}
    order = [start]
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph[current]):
            if neighbor not in subset_nodes or neighbor in visited:
                continue
            visited.add(neighbor)
            order.append(neighbor)
            queue.append(neighbor)
    if visited != subset_nodes:
        return None
    return order


def _select_connected_subset(backend: Any, graph: dict[int, set[int]], n_qubits: int) -> LayoutSelection:
    candidates: list[LayoutSelection] = []
    nodes = sorted(graph)
    for subset in combinations(nodes, n_qubits):
        order = _connected_subset_order(graph, subset)
        if order is None:
            continue
        error = 0.0
        coherence = 0.0
        for left, right in zip(order, order[1:]):
            error += _gate_error(backend, "cz", (left, right))
        for qubit in order:
            coherence += _qubit_coherence_score(backend, qubit)
        candidates.append(
            LayoutSelection(
                logical_to_physical=tuple(order),
                ring_supported=False,
                degraded=True,
                cz_error_score=float(error),
                coherence_score=float(coherence),
                reason="No simple ring found; using best connected subset with routing.",
            )
        )
    if not candidates:
        raise ValueError(f"Unable to find a connected {n_qubits}-qubit subset on the backend.")
    return sorted(candidates, key=lambda item: (item.cz_error_score, -item.coherence_score, item.logical_to_physical))[0]


def select_garnet_layout(backend: Any, n_qubits: int, policy: str = "dynamic") -> LayoutSelection:
    if policy != "dynamic":
        raise ValueError(f"Unsupported layout policy {policy!r}; only 'dynamic' is implemented.")
    edges = _extract_coupling_edges(backend)
    graph = _undirected_graph(edges)
    cycles = _find_simple_cycles(graph, n_qubits)
    if cycles:
        ranked: list[LayoutSelection] = []
        for cycle in cycles:
            error, coherence = _cycle_score(backend, cycle)
            ranked.append(
                LayoutSelection(
                    logical_to_physical=tuple(cycle),
                    ring_supported=True,
                    degraded=False,
                    cz_error_score=error,
                    coherence_score=coherence,
                    reason="Selected a simple ring with the lowest aggregate CZ error score.",
                )
            )
        return sorted(ranked, key=lambda item: (item.cz_error_score, -item.coherence_score, item.logical_to_physical))[0]
    return _select_connected_subset(backend, graph, n_qubits)


def resolve_iqm_backend(backend_mode: str, backend_url: str = DEFAULT_BACKEND_URL, backend_alias: str = DEFAULT_BACKEND_ALIAS) -> Any:
    IQMFakeGarnet, IQMProvider = import_iqm()
    normalized_mode = backend_mode.strip().lower()
    if normalized_mode == "fake":
        return IQMFakeGarnet()
    if normalized_mode == "mock":
        provider = IQMProvider(backend_url, quantum_computer=f"{backend_alias}:mock")
        return provider.get_backend()
    if normalized_mode == "facade":
        provider = IQMProvider(backend_url, quantum_computer=f"{backend_alias}:mock")
        return provider.get_backend(f"facade_{backend_alias}")
    if normalized_mode == "real":
        provider = IQMProvider(backend_url, quantum_computer=backend_alias)
        return provider.get_backend()
    raise ValueError(f"Unsupported backend_mode {backend_mode!r}.")


def transpile_circuits(
    circuits: list[Any],
    backend: Any,
    physical_layout: Optional[tuple[int, ...]] = None,
    optimization_level: int = 1,
) -> list[Any]:
    transpile = import_qiskit_transpile()
    kwargs: dict[str, Any] = {"backend": backend, "optimization_level": int(optimization_level)}
    if physical_layout is not None:
        kwargs["initial_layout"] = list(physical_layout)
    transpiled = transpile(circuits, **kwargs)
    if isinstance(transpiled, list):
        return transpiled
    return [transpiled]
