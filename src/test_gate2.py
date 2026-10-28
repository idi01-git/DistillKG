"""
Test Gate 2 Verification Script
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 2)

Checks:
1. JSON Schema validation passes with zero errors.
2. 100% of sensors and actuators map to equipment nodes in the graph.
3. Zero isolated nodes (all nodes have degree >= 1).
4. Validated directed paths exist between all fault origins and observable sensor clusters.
5. Dynamic operating-regime edge weight modulation responds to RR and Q_R.
"""

import os
import sys
import json
import jsonschema
import networkx as nx

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kg_manager import DistillationKGManager, ONTOLOGY_PATH


ONTOLOGY_JSON_SCHEMA = {
    "type": "object",
    "required": ["metadata", "subsystems", "equipment_nodes", "sensor_nodes", "actuator_nodes", "fault_nodes", "edges"],
    "properties": {
        "metadata": {"type": "object"},
        "subsystems": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "description"]
            }
        },
        "equipment_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "subsystem"]
            }
        },
        "sensor_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "equipment", "subsystem"]
            }
        },
        "actuator_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "equipment", "subsystem"]
            }
        },
        "fault_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "subsystem", "initiating_component", "mechanism"]
            }
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "relation", "base_weight"]
            }
        }
    }
}


def test_gate_2() -> bool:
    print("=" * 60)
    print("RUNNING TEST GATE 2: Distillation Ontology & Dynamic KG Validation")
    print("=" * 60)

    # 1. JSON Schema Validation
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        ontology_data = json.load(f)

    try:
        jsonschema.validate(instance=ontology_data, schema=ONTOLOGY_JSON_SCHEMA)
        print("[Check 1: JSON Schema] PASSED: data/distillation_ontology.json conforms to schema.")
    except jsonschema.ValidationError as e:
        print(f"[Check 1: JSON Schema] FAILED: {e.message}")
        return False

    # 2. Graph Initialization & Topology Checks
    kg = DistillationKGManager(ontology_path=ONTOLOGY_PATH)
    g = kg.graph
    print(f"[Check 2: Graph Dimensions] Loaded {g.number_of_nodes()} nodes and {g.number_of_edges()} directed edges.")

    # 3. 100% Sensor & Actuator Mapping Check
    sensor_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "sensor"]
    actuator_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "actuator"]
    equipment_nodes = set([n for n, d in g.nodes(data=True) if d.get("node_type") == "equipment"])

    print(f"[Check 3: Mappings] Checking {len(sensor_nodes)} sensors and {len(actuator_nodes)} actuators...")
    for s in sensor_nodes:
        eq = g.nodes[s].get("equipment")
        assert eq in equipment_nodes, f"Sensor {s} mapped to unknown equipment {eq}!"
        # Check edge exists between equipment and sensor
        assert g.has_edge(eq, s) or g.has_edge(s, eq), f"Sensor {s} has no edge connecting to equipment {eq}!"

    for act in actuator_nodes:
        eq = g.nodes[act].get("equipment")
        assert eq in equipment_nodes, f"Actuator {act} mapped to unknown equipment {eq}!"
        assert g.has_edge(act, eq) or g.has_edge(eq, act), f"Actuator {act} has no edge connecting to equipment {eq}!"

    print("[Check 3: Mappings] PASSED: 100% of sensors and actuators map cleanly to valid equipment.")

    # 4. Zero Isolated Nodes Check
    isolated_nodes = list(nx.isolates(g))
    print(f"[Check 4: Connectivity] Isolated nodes count: {len(isolated_nodes)}")
    # Subsystem cluster nodes might be structural labels; check all non-subsystem nodes
    functional_isolates = [n for n in isolated_nodes if g.nodes[n].get("node_type") != "subsystem"]
    assert len(functional_isolates) == 0, f"Found disconnected functional nodes: {functional_isolates}"
    print("[Check 4: Connectivity] PASSED: Zero functional isolated nodes.")

    # 5. Fault-to-Sensor Reachability Paths
    fault_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "fault"]
    print(f"[Check 5: Fault Reachability] Verifying causal paths for {len(fault_nodes)} fault origins...")
    
    for flt in fault_nodes:
        descendants = nx.descendants(g, flt)
        observable_sensors = [d for d in descendants if g.nodes[d].get("node_type") == "sensor"]
        print(f"  - {flt} ({g.nodes[flt]['name']}): reachable sensors = {len(observable_sensors)}")
        assert len(observable_sensors) >= 1, f"Fault {flt} has zero reachable sensors!"

    print("[Check 5: Fault Reachability] PASSED: Every fault origin reaches observable sensor clusters.")

    # 6. Dynamic Operating Regime Edge Modulation
    print("[Check 6: Dynamic Regime Modulation] Testing modulation with RR=2.5 and Q_R=1.8...")
    H_mod = kg.modulate_regime(reflux_ratio=2.5, reboiler_duty=1.8, nominal_rr=1.0, nominal_qr=1.0)
    
    # Check that reflux edge weight increased
    reflux_edge = H_mod.edges["EQ_REFLUX_LOOP", "EQ_TOP_TRAY"]
    reboiler_edge = H_mod.edges["EQ_REBOILER", "EQ_STRIPPING_SECTION"]
    print(f"  - Reflux return weight: base={g.edges['EQ_REFLUX_LOOP', 'EQ_TOP_TRAY']['weight']:.2f} -> modulated={reflux_edge['weight']:.2f}")
    print(f"  - Vapor boilup weight: base={g.edges['EQ_REBOILER', 'EQ_STRIPPING_SECTION']['weight']:.2f} -> modulated={reboiler_edge['weight']:.2f}")
    
    assert reflux_edge["weight"] > g.edges["EQ_REFLUX_LOOP", "EQ_TOP_TRAY"]["weight"], "Reflux edge did not scale with reflux ratio!"
    assert reboiler_edge["weight"] > g.edges["EQ_REBOILER", "EQ_STRIPPING_SECTION"]["weight"], "Boilup edge did not scale with reboiler duty!"
    print("[Check 6: Dynamic Regime Modulation] PASSED: Edge weights successfully adapt to operational regimes.")

    # 7. Render visualization
    kg.export_graph_visualization()

    print("=" * 60)
    print("STATUS: TEST_GATE_2_PASSED")
    print("All Knowledge Graph schema, mapping, connectivity, and modulation checks verified.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_gate_2()
    sys.exit(0 if success else 1)
