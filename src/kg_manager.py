"""
Knowledge Graph Manager & Dynamic Regime Modulator
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 2)

Constructs directed distillation P&ID graph in networkx, modulates edge weights
dynamically based on reflux ratio RR(t) and reboiler heat duty Q_R(t), and
provides topological precedence querying for root-cause isolation.
"""

import os
import sys
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Set, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ONTOLOGY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "distillation_ontology.json")
OUTPUT_PLOT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "dynamic_kg_schema.png")


class DistillationKGManager:
    def __init__(self, ontology_path: str = ONTOLOGY_PATH):
        self.ontology_path = ontology_path
        self.graph = nx.DiGraph()
        self.raw_ontology = self._load_ontology()
        self._build_graph()

    def _load_ontology(self) -> Dict[str, Any]:
        if not os.path.exists(self.ontology_path):
            raise FileNotFoundError(f"Ontology file not found at {self.ontology_path}")
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _build_graph(self):
        """Constructs the NetworkX directed graph with node attributes."""
        # Add Subsystem clusters
        for sub in self.raw_ontology.get("subsystems", []):
            self.graph.add_node(sub["id"], node_type="subsystem", name=sub["name"])

        # Add Equipment nodes
        for eq in self.raw_ontology.get("equipment_nodes", []):
            self.graph.add_node(
                eq["id"],
                node_type="equipment",
                subsystem=eq["subsystem"],
                name=eq["name"]
            )

        # Add Sensor nodes
        for s in self.raw_ontology.get("sensor_nodes", []):
            self.graph.add_node(
                s["id"],
                node_type="sensor",
                equipment=s["equipment"],
                subsystem=s["subsystem"],
                sensor_type=s.get("type", "process"),
                name=s["name"]
            )

        # Add Actuator nodes
        for act in self.raw_ontology.get("actuator_nodes", []):
            self.graph.add_node(
                act["id"],
                node_type="actuator",
                equipment=act["equipment"],
                subsystem=act["subsystem"],
                name=act["name"]
            )

        # Add Fault nodes
        for flt in self.raw_ontology.get("fault_nodes", []):
            self.graph.add_node(
                flt["id"],
                node_type="fault",
                initiating_component=flt["initiating_component"],
                subsystem=flt["subsystem"],
                name=flt["name"]
            )

        # Add Edges
        for edge in self.raw_ontology.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                relation=edge["relation"],
                base_weight=float(edge.get("base_weight", 1.0)),
                weight=float(edge.get("base_weight", 1.0)),
                dynamic_regime=bool(edge.get("dynamic_regime", False)),
                regime_driver=edge.get("regime_driver", "none")
            )

    def modulate_regime(
        self,
        reflux_ratio: float,
        reboiler_duty: float,
        nominal_rr: float = 1.0,
        nominal_qr: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> nx.DiGraph:
        """
        Dynamically updates edge weights based on operating conditions:
        W_ij(t) = W_0 * (1 + alpha * |RR(t) - RR_0| / RR_0) for liquid downflow
        W_ij(t) = W_0 * (1 + beta * |Q_R(t) - Q_R0| / Q_R0) for vapor boilup
        """
        H = self.graph.copy()
        
        rr_factor = 1.0 + alpha * (abs(reflux_ratio - nominal_rr) / max(nominal_rr, 1e-4))
        qr_factor = 1.0 + beta * (abs(reboiler_duty - nominal_qr) / max(nominal_qr, 1e-4))

        for u, v, d in H.edges(data=True):
            if d.get("dynamic_regime", False):
                driver = d.get("regime_driver", "")
                base = d.get("base_weight", 1.0)
                if driver == "reflux_ratio":
                    d["weight"] = base * rr_factor
                elif driver == "reboiler_duty":
                    d["weight"] = base * qr_factor

        return H

    def is_ancestor(self, ancestor_node: str, descendant_node: str) -> bool:
        """Checks if ancestor_node has a directed path to descendant_node."""
        if not self.graph.has_node(ancestor_node) or not self.graph.has_node(descendant_node):
            return False
        return nx.has_path(self.graph, ancestor_node, descendant_node)

    def get_ancestors(self, node: str) -> Set[str]:
        if not self.graph.has_node(node):
            return set()
        return nx.ancestors(self.graph, node)

    def get_descendants(self, node: str) -> Set[str]:
        if not self.graph.has_node(node):
            return set()
        return nx.descendants(self.graph, node)

    def prune_symptoms_to_root(self, candidate_nodes: List[str]) -> List[str]:
        """
        Two-Tier NeuroSymbolic Intersection:
        Given candidate variables flagged as anomalous, prune any variable that is
        a downstream reachable descendant of another active candidate.
        Returns the minimal ancestral root causes.
        """
        candidates = set(candidate_nodes)
        roots = []
        for cand in candidates:
            # Check if any other candidate is an ancestor of cand
            has_candidate_ancestor = False
            cand_ancestors = self.get_ancestors(cand)
            for other in candidates:
                if other != cand and other in cand_ancestors:
                    has_candidate_ancestor = True
                    break
            if not has_candidate_ancestor:
                roots.append(cand)
        return roots if roots else list(candidates)

    def export_graph_visualization(self, save_path: str = OUTPUT_PLOT_PATH):
        """Renders and saves a clean publication-grade topology plot."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(14, 10))

        # Color nodes by type
        color_map = {
            "fault": "#ff4d4f",
            "equipment": "#1890ff",
            "sensor": "#52c41a",
            "actuator": "#faad14",
            "subsystem": "#722ed1"
        }
        
        node_colors = []
        for n in self.graph.nodes():
            ntype = self.graph.nodes[n].get("node_type", "equipment")
            node_colors.append(color_map.get(ntype, "#d9d9d9"))

        # Use spring layout with fixed seed
        pos = nx.spring_layout(self.graph, k=0.35, iterations=60, seed=42)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=320, alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, edge_color="#bfbfbf", arrows=True, arrowsize=12, width=1.0)
        
        # Draw labels only for equipment and faults to prevent clutter
        labels = {
            n: n for n in self.graph.nodes()
            if self.graph.nodes[n].get("node_type") in ["fault", "equipment"]
        }
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8, font_family="sans-serif")

        plt.title("Distillation Column P&ID Knowledge Graph: Physical Topology & Dynamic Causal Paths", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Graph schema visualization saved to {save_path}")


if __name__ == "__main__":
    kg = DistillationKGManager()
    print(f"Graph initialized: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
    kg.export_graph_visualization()
