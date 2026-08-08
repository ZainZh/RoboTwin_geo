"""Small, dependency-free subset of ``partnext.object``."""

from __future__ import annotations

import trimesh


def scene2meshes(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return [scene_or_mesh]
    if isinstance(scene_or_mesh, trimesh.Scene):
        result = []
        for node_name in scene_or_mesh.graph.nodes_geometry:
            transform, geometry_name = scene_or_mesh.graph[node_name]
            geometry = scene_or_mesh.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            result.append(geometry)
        return result
    raise TypeError(f"unsupported mesh container: {type(scene_or_mesh)!r}")
