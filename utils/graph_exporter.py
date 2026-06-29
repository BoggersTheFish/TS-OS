import json
import os
from typing import Dict, Any

class GraphExporter:
    """
    Layer 4: Reasoning Substrate Graph Exporter
    Serializes the discrete graph structure into a strictly formatted JSON file.
    Translates NumPy data types and Python tuples into standard JSON structures.
    """
    @staticmethod
    def serialize_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts graph structures to JSON-serializable standard Python formats.
        """
        serializable_nodes = []
        for node in graph['nodes']:
            serializable_nodes.append({
                'id': int(node['id']),
                'pos': [float(node['pos'][0]), float(node['pos'][1])],
                'grid_idx': [int(node['grid_idx'][0]), int(node['grid_idx'][1])],
                'intensity': float(node['intensity']),
                'curvature': [float(node['curvature'][0]), float(node['curvature'][1])]
            })

        serializable_edges = []
        for edge in graph['edges']:
            serializable_edges.append({
                'id': int(edge['id']),
                'source': int(edge['source']),
                'target': int(edge['target']),
                'saddle_pos': [float(edge['saddle_pos'][0]), float(edge['saddle_pos'][1])],
                'tension': float(edge['tension']),
                'path': [[float(p[0]), float(p[1])] for p in edge['path']]
            })

        serializable_saddles = []
        for saddle in graph.get('saddles', []):
            serializable_saddles.append({
                'id': int(saddle['id']),
                'pos': [float(saddle['pos'][0]), float(saddle['pos'][1])],
                'grid_idx': [int(saddle['grid_idx'][0]), int(saddle['grid_idx'][1])],
                'grad_mag': float(saddle['grad_mag']),
                'curvature': [float(saddle['curvature'][0]), float(saddle['curvature'][1])]
            })

        return {
            'nodes': serializable_nodes,
            'edges': serializable_edges,
            'saddles': serializable_saddles
        }

    @classmethod
    def export_to_json(cls, graph: Dict[str, Any], filepath: str) -> None:
        """
        Serializes and writes the compiled topological graph to a JSON file.
        Creates any parent directories if they don't exist.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Serialize the graph to JSON format
        serialized = cls.serialize_graph(graph)
        
        with open(filepath, 'w') as f:
            json.dump(serialized, f, indent=2)
        
        print(f"[GraphExporter] Successfully serialized topological skeleton to {filepath}")
        print(f"                Nodes: {len(serialized['nodes'])} | Edges: {len(serialized['edges'])} | Saddles: {len(serialized['saddles'])}")
