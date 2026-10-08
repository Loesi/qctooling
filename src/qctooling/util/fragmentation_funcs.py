from __future__ import annotations
from collections import deque
from ase.data import atomic_numbers, covalent_radii
import numpy as np
import numpy.typing as npt
from typing import List
from ..classes import Xyz
from .conversions import au2ang

def bondMatrix(xyz: Xyz) -> npt.NDArray[np.bool]:
    coords = au2ang(xyz.coordinates)
    dist = np.linalg.norm(coords[:, None] - coords, axis=-1)
    radii = covalent_radii[np.array([atomic_numbers[s] for s in xyz.elements])]
    bonded = (dist < ((radii[:,None] + radii[None,:]) * 1.25))
    np.fill_diagonal(bonded, 0)
    return bonded

def fragment_noML(matrix: npt.NDArray[np.bool], elements: npt.NDArray[np.str_]) -> npt.NDArray[np.bool]:
    for i, e in enumerate(elements):
        if e in ["Fe", "Mn", "Co", "Ru", "Pt"]:
            matrix[i] = 0
            matrix[:,i] = 0
    return matrix

def fragment_specific_cut(matrix: npt.NDArray[np.bool], cut_str: str) -> npt.NDArray[np.bool]:
    cut_pos = [(int(i), int(j)) for bond in cut_str.split(",") for i,j in bond.split("-")]
    for i, j in cut_pos:
        if not matrix[i,j] or not matrix[j,i]:
            raise ValueError("There is no bond between atoms {i} and {j}")
        matrix[i,j] = matrix[j,i] = 0
    return matrix

def _is_bridge(neighbors: list[list[int]], a: int, b: int) -> bool:
    seen = {a}
    queue = deque([a])
    while queue:
        node = queue.popleft()
        for nxt in neighbors[node]:
            if node == a and nxt == b:
                continue  # skip the bond under test
            if nxt == b:
                return False  # b reachable another way -> bond is in a cycle
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return True

def _find_ring_bonds(neighbors: list[list[int]]) -> list[tuple[int, int]]:
    ring_bonds = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i < j and not _is_bridge(neighbors, i, j):
                ring_bonds.append((i,j))
    return ring_bonds

def _connected_components(nodes: set[int], neighbors: list[list[int]]) -> list[set[int]]:
    """Connected components of the subgraph induced by `nodes`."""
    remaining = set(nodes)
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in neighbors[node]:
                if nxt in remaining:
                    remaining.discard(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components

def fragment_rings(bonds: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:

    bonds = np.asarray(bonds, dtype=bool)
    if bonds.ndim != 2 or bonds.shape[0] != bonds.shape[1]:
        raise ValueError("bond array must be square (N, N)")

    adj = bonds | bonds.T
    np.fill_diagonal(adj, False)
    neighbors = [np.flatnonzero(row).tolist() for row in adj]
    n = len(neighbors)

    # Ring-only adjacency -> ring atoms and ring systems
    ring_neighbors: list[list[int]] = [[] for _ in range(n)]
    for i, j in _find_ring_bonds(neighbors):
        ring_neighbors[i].append(j)
        ring_neighbors[j].append(i)
    ring_atoms = {i for i, nbrs in enumerate(ring_neighbors) if nbrs}

    systems = _connected_components(ring_atoms, ring_neighbors)
    system_of: dict[int, int] = {
        atom: idx for idx, system in enumerate(systems) for atom in system
    }

    cuts: list[tuple[int, int]] = []

    # Direct links: acyclic bond between atoms of two different ring systems
    for i in ring_atoms:
        for j in neighbors[i]:
            if i < j and j in ring_atoms and system_of[i] != system_of[j]:
                cuts.append((i, j))

    # Chain links: acyclic group touching two or more ring systems
    non_ring_atoms = set(range(n)) - ring_atoms
    for group in _connected_components(non_ring_atoms, neighbors):
        attachments = [
            (atom, nbr)
            for atom in group
            for nbr in neighbors[atom]
            if nbr in ring_atoms
        ]
        if len({system_of[nbr] for _, nbr in attachments}) >= 2:
            cuts.extend(attachments)

    result = bonds.copy()
    for i, j in cuts:
        result[i, j] = False
        result[j, i] = False
    return result

def fragment_mergeLinker(bonds: npt.NDArray[np.bool_], pure_bond: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    adjacency = [np.flatnonzero(row).tolist() for row in bonds]
    pure_adjacency = [np.flatnonzero(row).tolist() for row in pure_bond]
    components = _connected_components(set(range(len(bonds))), adjacency)
    component_of = {atom: idx for idx, component in enumerate(components) for atom in component}

    for linker in components:
        linker_edges = sum(
            neighbor in linker
            for atom in linker
            for neighbor in pure_adjacency[atom]
            ) // 2
        if linker_edges != len(linker) - 1:
            continue

        attached_components = {
            component_of[neighbor]
            for atom in linker
            for neighbor in pure_adjacency[atom]
            if neighbor not in linker
        }
        if len(attached_components) < 2:
            continue

        target_idx = max(attached_components, key=lambda idx: len(components[idx]))
        target = components[target_idx]
        for atom in linker:
            for neighbor in pure_adjacency[atom]:
                if neighbor in target:
                    bonds[atom, neighbor] = True
                    bonds[neighbor, atom] = True

    return bonds

def convert_bondMat2List(matrix: npt.NDArray[np.bool]) -> List[List[int]]:
    fragments = []
    assigned = np.zeros(matrix.shape[0], np.bool)

    for idx in range(len(matrix)):
        if assigned[idx]:
            continue
        frag, stack = [], [idx]
        assigned[idx] = 1
        while stack:
            current = stack.pop()
            frag.append(current)
            for neighbor in np.flatnonzero(matrix[current]):
                if not assigned[neighbor]:
                    assigned[neighbor] = 1
                    stack.append(int(neighbor))
        fragments.append(sorted(frag))

    return sorted(fragments, key=len)

def fragmentate_ligands(xyz: Xyz, cut_str: str = "") -> List[List[int]]:
    bondMat = bondMatrix(xyz)
    if cut_str != "":
        bondMat = fragment_specific_cut(bondMat, cut_str)
    bondMat = fragment_noML(bondMat, xyz.elements)
    return convert_bondMat2List(bondMat)

def fragmentate_rings(xyz: Xyz, cut_str: str = "") -> List[List[int]]:
    bondMat = bondMatrix(xyz)
    if cut_str != "":
        bondMat = fragment_specific_cut(bondMat, cut_str)
    bondMat = fragment_noML(bondMat, xyz.elements)
    before_rings = np.copy(bondMat)
    bondMat = fragment_rings(bondMat)
    bondMat = fragment_mergeLinker(bondMat, before_rings)
    return convert_bondMat2List(bondMat)
