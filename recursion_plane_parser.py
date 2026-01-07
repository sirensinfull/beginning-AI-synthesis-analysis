"""
Recursion Plane Symbolic Parser
Translates 2D symbolic diagrams into computational structures
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    GROWTH = ">"
    MITOSIS = "<"
    NEXUS = "x"
    HARD_STOP = "."
    SOFT_STOP = ":"
    PERFECT_LOOP = "8"
    INCISED_LOOP = "9"
    HEALED_LOOP = "10"
    PATH_H = "_"
    PATH_D = "\\"
    PATH_V = "/"

@dataclass
class SymbolicNode:
    x: int
    y: int
    symbol: str
    node_type: NodeType
    energy_level: float = 1.0
    connections: List['SymbolicNode'] = None
    dimensional_phase: int = 0
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []

@dataclass
class DimensionalCrossover:
    point: Tuple[int, int]
    entry_node: SymbolicNode
    exit_node: SymbolicNode
    crossover_type: str
    energy_transfer: float

@dataclass
class RecursiveLoop:
    nodes: List[SymbolicNode]
    loop_type: NodeType
    stability: float
    energy_flow: float
    dimensional_integrity: bool

class RecursionPlaneParser:
    """
    Parses symbolic recursion plane diagrams and extracts all possibilities
    """
    
    def __init__(self):
        self.nodes: Dict[Tuple[int, int], SymbolicNode] = {}
        self.loops: List[RecursiveLoop] = []
        self.crossovers: List[DimensionalCrossover] = []
        self.dimensional_map: Dict[int, Set[SymbolicNode]] = {}
        
    def parse_diagram(self, diagram: str) -> Dict[str, any]:
        """
        Parse a symbolic diagram and extract all structural information
        
        Args:
            diagram: Multi-line string containing symbolic representation
            
        Returns:
            Complete analysis of the recursion plane structure
        """
        lines = diagram.strip().split('\n')
        
        # Phase 1: Extract all symbols and their positions
        self._extract_symbols(lines)
        
        # Phase 2: Identify connections and pathways
        self._map_connections()
        
        # Phase 3: Detect loops and their states
        self._identify_loops()
        
        # Phase 4: Find dimensional crossovers
        self._identify_crossovers()
        
        # Phase 5: Analyze all possibilities
        analysis = self._analyze_possibilities()
        
        return analysis
    
    def _extract_symbols(self, lines: List[str]):
        """Extract all symbols from the diagram grid"""
        multi_char_symbols = ['10', '8', '9']  # Priority for multi-character
        
        y = 0
        for line in lines:
            x = 0
            while x < len(line):
                symbol = line[x]
                
                # Check for multi-character symbols
                if x + 1 < len(line):
                    potential_multi = line[x:x+2]
                    if potential_multi in multi_char_symbols:
                        symbol = potential_multi
                        node_type = self._get_node_type(symbol)
                        self.nodes[(x, y)] = SymbolicNode(x, y, symbol, node_type)
                        x += 2
                        continue
                
                # Single character symbol
                if symbol.strip() and symbol != ' ':
                    node_type = self._get_node_type(symbol)
                    self.nodes[(x, y)] = SymbolicNode(x, y, symbol, node_type)
                
                x += 1
            y += 1
    
    def _get_node_type(self, symbol: str) -> NodeType:
        """Map symbol to node type"""
        symbol_to_type = {
            '>': NodeType.GROWTH,
            '<': NodeType.MITOSIS,
            'x': NodeType.NEXUS,
            '.': NodeType.HARD_STOP,
            ':': NodeType.SOFT_STOP,
            '8': NodeType.PERFECT_LOOP,
            '9': NodeType.INCISED_LOOP,
            '10': NodeType.HEALED_LOOP,
            '_': NodeType.PATH_H,
            '\\': NodeType.PATH_D,
            '/': NodeType.PATH_V
        }
        return symbol_to_type.get(symbol, NodeType.PATH_H)
    
    def _map_connections(self):
        """Identify connections between nodes based on proximity and pathways"""
        for pos, node in self.nodes.items():
            x, y = pos
            
            # Define search patterns based on node type
            if node.node_type == NodeType.GROWTH:
                # Growth nodes connect forward (right, down-right)
                connections = self._find_connections_in_direction(x, y, [(1, 0), (1, 1), (0, 1)])
            elif node.node_type == NodeType.MITOSIS:
                # Mitosis nodes connect backward (left, up-left)  
                connections = self._find_connections_in_direction(x, y, [(-1, 0), (-1, -1), (0, -1)])
            elif node.node_type == NodeType.NEXUS:
                # Nexus connects in all directions
                connections = self._find_connections_in_direction(x, y, 
                    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)])
            else:
                # Pathways connect along their axis
                connections = self._find_pathway_connections(x, y, node)
            
            node.connections = connections
    
    def _find_connections_in_direction(self, x: int, y: int, directions: List[Tuple[int, int]]) -> List[SymbolicNode]:
        """Find valid connections in specified directions"""
        connections = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) in self.nodes:
                connections.append(self.nodes[(new_x, new_y)])
        return connections
    
    def _find_pathway_connections(self, x: int, y: int, node: SymbolicNode) -> List[SymbolicNode]:
        """Find connections for pathway nodes"""
        if node.node_type == NodeType.PATH_H:
            directions = [(1, 0), (-1, 0)]  # Horizontal
        elif node.node_type == NodeType.PATH_V:
            directions = [(0, 1), (0, -1)]  # Vertical
        elif node.node_type == NodeType.PATH_D:
            directions = [(1, 1), (-1, -1)]  # Diagonal
        else:
            directions = []
        
        return self._find_connections_in_direction(x, y, directions)
    
    def _identify_loops(self):
        """Identify recursive loops and their states"""
        visited = set()
        
        for pos, node in self.nodes.items():
            if pos in visited:
                continue
                
            if node.node_type in [NodeType.PERFECT_LOOP, NodeType.INCISED_LOOP, NodeType.HEALED_LOOP]:
                loop = self._trace_loop(node, visited)
                if loop:
                    self.loops.append(loop)
    
    def _trace_loop(self, start_node: SymbolicNode, visited: Set[Tuple[int, int]]) -> Optional[RecursiveLoop]:
        """Trace a complete loop starting from a loop node"""
        loop_nodes = [start_node]
        current = start_node
        loop_visited = {current.x, current.y}
        
        # Follow connections to complete the loop
        for _ in range(100):  # Prevent infinite loops
            next_nodes = [n for n in current.connections 
                         if (n.x, n.y) not in loop_visited and 
                         n.node_type in [NodeType.GROWTH, NodeType.MITOSIS, NodeType.NEXUS, 
                                       NodeType.PERFECT_LOOP, NodeType.INCISED_LOOP, NodeType.HEALED_LOOP]]
            
            if not next_nodes:
                break
                
            current = next_nodes[0]
            loop_nodes.append(current)
            loop_visited.add((current.x, current.y))
            
            # Check if we completed the loop
            if current == start_node and len(loop_nodes) > 2:
                stability = self._calculate_loop_stability(loop_nodes)
                energy_flow = self._calculate_energy_flow(loop_nodes)
                dimensional_integrity = self._check_dimensional_integrity(loop_nodes)
                
                return RecursiveLoop(
                    nodes=loop_nodes,
                    loop_type=start_node.node_type,
                    stability=stability,
                    energy_flow=energy_flow,
                    dimensional_integrity=dimensional_integrity
                )
        
        return None
    
    def _calculate_loop_stability(self, nodes: List[SymbolicNode]) -> float:
        """Calculate the stability of a recursive loop"""
        if not nodes:
            return 0.0
        
        # Base stability on node types
        stability_scores = {
            NodeType.PERFECT_LOOP: 1.0,
            NodeType.HEALED_LOOP: 0.8,
            NodeType.INCISED_LOOP: 0.3,
            NodeType.GROWTH: 0.7,
            NodeType.MITOSIS: 0.7,
            NodeType.NEXUS: 0.9
        }
        
        total_stability = sum(stability_scores.get(node.node_type, 0.5) for node in nodes)
        return total_stability / len(nodes)
    
    def _calculate_energy_flow(self, nodes: List[SymbolicNode]) -> float:
        """Calculate energy flow in the loop"""
        # Energy flow is based on connections and node types
        flow = 0.0
        for i in range(len(nodes)):
            current = nodes[i]
            next_node = nodes[(i + 1) % len(nodes)]
            
            if next_node in current.connections:
                # Valid connection contributes to flow
                if current.node_type == NodeType.GROWTH:
                    flow += 1.2
                elif current.node_type == NodeType.MITOSIS:
                    flow += 0.8
                else:
                    flow += 1.0
        
        return flow / len(nodes) if nodes else 0.0
    
    def _check_dimensional_integrity(self, nodes: List[SymbolicNode]) -> bool:
        """Check if loop maintains dimensional consistency"""
        if not nodes:
            return False
        
        # All nodes in a stable loop should be in compatible dimensional phases
        phases = [node.dimensional_phase for node in nodes]
        return len(set(phases)) <= 2  # Allow for dimensional crossover
    
    def _identify_crossovers(self):
        """Identify dimensional crossover points"""
        # Look for nexus nodes between different dimensional phases
        for pos, node in self.nodes.items():
            if node.node_type == NodeType.NEXUS:
                # Check if this nexus connects different dimensional regions
                crossover = self._analyze_nexus_crossover(node)
                if crossover:
                    self.crossovers.append(crossover)
    
    def _analyze_nexus_crossover(self, nexus: SymbolicNode) -> Optional[DimensionalCrossover]:
        """Analyze if a nexus creates a dimensional crossover"""
        if len(nexus.connections) < 2:
            return None
        
        # Look for connections that span different node types or energy levels
        growth_connections = [c for c in nexus.connections if c.node_type == NodeType.GROWTH]
        mitosis_connections = [c for c in nexus.connections if c.node_type == NodeType.MITOSIS]
        
        if growth_connections and mitosis_connections:
            # This is a dimensional crossover point
            return DimensionalCrossover(
                point=(nexus.x, nexus.y),
                entry_node=growth_connections[0],
                exit_node=mitosis_connections[0],
                crossover_type="Growth-Mitosis Bridge",
                energy_transfer=0.5  # Default transfer coefficient
            )
        
        return None
    
    def _analyze_possibilities(self) -> Dict[str, any]:
        """Analyze all possible states and transformations"""
        analysis = {
            "total_nodes": len(self.nodes),
            "node_distribution": self._get_node_distribution(),
            "total_loops": len(self.loops),
            "loop_states": self._get_loop_states(),
            "total_crossovers": len(self.crossovers),
            "crossover_types": [c.crossover_type for c in self.crossovers],
            "ankh_repair_opportunities": self._identify_ankh_opportunities(),
            "dimensional_topology": self._analyze_dimensional_topology(),
            "energy_distribution": self._analyze_energy_distribution(),
            "transformation_sequences": self._identify_transformation_sequences(),
            "stability_analysis": self._analyze_stability(),
            "growth_potential": self._calculate_growth_potential(),
            "mitosis_potential": self._calculate_mitosis_potential()
        }
        
        return analysis
    
    def _get_node_distribution(self) -> Dict[str, int]:
        """Get distribution of node types"""
        distribution = {}
        for node in self.nodes.values():
            node_type = node.node_type.name
            distribution[node_type] = distribution.get(node_type, 0) + 1
        return distribution
    
    def _get_loop_states(self) -> Dict[str, int]:
        """Get count of different loop states"""
        states = {"perfect": 0, "incised": 0, "healed": 0}
        for loop in self.loops:
            if loop.loop_type == NodeType.PERFECT_LOOP:
                states["perfect"] += 1
            elif loop.loop_type == NodeType.INCISED_LOOP:
                states["incised"] += 1
            elif loop.loop_type == NodeType.HEALED_LOOP:
                states["healed"] += 1
        return states
    
    def _identify_ankh_opportunities(self) -> List[Dict[str, any]]:
        """Identify opportunities for Ankh repair processes"""
        opportunities = []
        
        for loop in self.loops:
            if loop.loop_type == NodeType.INCISED_LOOP:
                # Find nearby nexus points that could serve as repair bridges
                nearby_nexus = self._find_nearby_nexus(loop.nodes)
                
                if nearby_nexus:
                    opportunities.append({
                        "damaged_loop": [(n.x, n.y) for n in loop.nodes],
                        "repair_bridge": (nearby_nexus.x, nearby_nexus.y),
                        "estimated_stability_gain": 0.5,  # Placeholder
                        "energy_flow_restoration": 0.7   # Placeholder
                    })
        
        return opportunities
    
    def _find_nearby_nexus(self, loop_nodes: List[SymbolicNode]) -> Optional[SymbolicNode]:
        """Find nexus nodes near a loop that could serve as repair points"""
        loop_positions = {(n.x, n.y) for n in loop_nodes}
        
        for pos, node in self.nodes.items():
            if node.node_type == NodeType.NEXUS and pos not in loop_positions:
                # Check if this nexus is adjacent to the loop
                for loop_pos in loop_positions:
                    distance = abs(pos[0] - loop_pos[0]) + abs(pos[1] - loop_pos[1])
                    if distance <= 2:  # Within repair range
                        return node
        
        return None
    
    def _analyze_dimensional_topology(self) -> Dict[str, any]:
        """Analyze the dimensional structure of the recursion plane"""
        # Group nodes by dimensional affinity
        dimensional_groups = {}
        
        for node in self.nodes.values():
            # Assign dimensional phase based on position and connections
            phase = self._calculate_dimensional_phase(node)
            node.dimensional_phase = phase
            
            if phase not in dimensional_groups:
                dimensional_groups[phase] = []
            dimensional_groups[phase].append((node.x, node.y, node.symbol))
        
        return {
            "phase_count": len(dimensional_groups),
            "phase_distribution": dimensional_groups,
            "crossover_points": len(self.crossovers),
            "interdimensional_connections": len([c for c in self.crossovers if c.energy_transfer > 0])
        }
    
    def _calculate_dimensional_phase(self, node: SymbolicNode) -> int:
        """Calculate the dimensional phase of a node"""
        # Simple heuristic: phase based on position modulo prime numbers
        return (node.x + node.y) % 7  # 7-phase dimensional system
    
    def _analyze_energy_distribution(self) -> Dict[str, float]:
        """Analyze energy distribution across the plane"""
        total_energy = 0.0
        node_energies = {}
        
        for node in self.nodes.values():
            energy = self._calculate_node_energy(node)
            node_energies[(node.x, node.y)] = energy
            total_energy += energy
        
        return {
            "total_energy": total_energy,
            "average_energy": total_energy / len(self.nodes) if self.nodes else 0,
            "energy_by_node": node_energies,
            "energy_concentration": self._calculate_energy_concentration(node_energies)
        }
    
    def _calculate_node_energy(self, node: SymbolicNode) -> float:
        """Calculate energy level of a single node"""
        base_energy = {
            NodeType.GROWTH: 1.2,
            NodeType.MITOSIS: 0.8,
            NodeType.NEXUS: 1.5,
            NodeType.PERFECT_LOOP: 2.0,
            NodeType.INCISED_LOOP: 0.5,
            NodeType.HEALED_LOOP: 1.8,
            NodeType.HARD_STOP: 0.1,
            NodeType.SOFT_STOP: 0.3,
            NodeType.PATH_H: 0.6,
            NodeType.PATH_D: 0.7,
            NodeType.PATH_V: 0.6
        }
        
        return base_energy.get(node.node_type, 0.5)
    
    def _calculate_energy_concentration(self, energies: Dict[Tuple[int, int], float]) -> float:
        """Calculate how concentrated energy is in the system"""
        if not energies:
            return 0.0
        
        values = list(energies.values())
        mean_energy = sum(values) / len(values)
        variance = sum((e - mean_energy) ** 2 for e in values) / len(values)
        
        return variance / (mean_energy ** 2) if mean_energy > 0 else 0.0
    
    def _identify_transformation_sequences(self) -> List[List[str]]:
        """Identify possible transformation sequences"""
        sequences = []
        
        # Ankh transformation: 9 -> x -> 10
        for loop in self.loops:
            if loop.loop_type == NodeType.INCISED_LOOP:
                # Find path through nexus to healed state
                nexus_path = self._find_path_to_nexus(loop.nodes[0])
                if nexus_path:
                    sequences.append(["incised_loop", "nexus_bridge", "healed_loop"])
        
        # Growth sequences: > -> x -> >
        growth_chains = self._find_growth_chains()
        for chain in growth_chains:
            sequences.append([">", "x", ">"])
        
        # Mitosis sequences: < -> x -> <
        mitosis_chains = self._find_mitosis_chains()
        for chain in mitosis_chains:
            sequences.append(["<", "x", "<"])
        
        return sequences
    
    def _find_path_to_nexus(self, start_node: SymbolicNode) -> Optional[List[SymbolicNode]]:
        """Find path from a node to nearest nexus"""
        # Simple BFS to find nexus
        visited = {(start_node.x, start_node.y)}
        queue = [(start_node, [start_node])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current.node_type == NodeType.NEXUS:
                return path
            
            for connection in current.connections:
                if (connection.x, connection.y) not in visited:
                    visited.add((connection.x, connection.y))
                    queue.append((connection, path + [connection]))
        
        return None
    
    def _find_growth_chains(self) -> List[List[SymbolicNode]]:
        """Find chains of growth nodes connected through nexus"""
        chains = []
        
        for node in self.nodes.values():
            if node.node_type == NodeType.GROWTH:
                # Look for growth -> nexus -> growth pattern
                for connection in node.connections:
                    if connection.node_type == NodeType.NEXUS:
                        for nexus_connection in connection.connections:
                            if (nexus_connection.node_type == NodeType.GROWTH and 
                                nexus_connection != node):
                                chains.append([node, connection, nexus_connection])
        
        return chains
    
    def _find_mitosis_chains(self) -> List[List[SymbolicNode]]:
        """Find chains of mitosis nodes connected through nexus"""
        chains = []
        
        for node in self.nodes.values():
            if node.node_type == NodeType.MITOSIS:
                # Look for mitosis -> nexus -> mitosis pattern
                for connection in node.connections:
                    if connection.node_type == NodeType.NEXUS:
                        for nexus_connection in connection.connections:
                            if (nexus_connection.node_type == NodeType.MITOSIS and 
                                nexus_connection != node):
                                chains.append([node, connection, nexus_connection])
        
        return chains
    
    def _analyze_stability(self) -> Dict[str, float]:
        """Analyze overall system stability"""
        loop_stability = sum(loop.stability for loop in self.loops) / len(self.loops) if self.loops else 1.0
        connection_density = self._calculate_connection_density()
        termination_balance = self._calculate_termination_balance()
        
        overall_stability = (loop_stability + connection_density + termination_balance) / 3.0
        
        return {
            "overall_stability": overall_stability,
            "loop_stability": loop_stability,
            "connection_density": connection_density,
            "termination_balance": termination_balance
        }
    
    def _calculate_connection_density(self) -> float:
        """Calculate how well connected the system is"""
        if not self.nodes:
            return 0.0
        
        total_possible_connections = len(self.nodes) * (len(self.nodes) - 1) / 2
        actual_connections = sum(len(node.connections) for node in self.nodes.values()) / 2
        
        return actual_connections / total_possible_connections if total_possible_connections > 0 else 0.0
    
    def _calculate_termination_balance(self) -> float:
        """Calculate balance between growth and termination"""
        growth_count = sum(1 for n in self.nodes.values() if n.node_type == NodeType.GROWTH)
        termination_count = sum(1 for n in self.nodes.values() 
                              if n.node_type in [NodeType.HARD_STOP, NodeType.SOFT_STOP])
        
        if growth_count == 0 and termination_count == 0:
            return 1.0
        
        # Perfect balance is when growth and termination are equal
        balance = 1.0 - abs(growth_count - termination_count) / (growth_count + termination_count + 1)
        return max(0.0, balance)
    
    def _calculate_growth_potential(self) -> float:
        """Calculate potential for system growth"""
        growth_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.GROWTH]
        nexus_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.NEXUS]
        
        if not growth_nodes:
            return 0.0
        
        # Growth potential increases with more growth nodes and available nexus points
        potential = len(growth_nodes) * 0.3 + len(nexus_nodes) * 0.2
        return min(1.0, potential)
    
    def _calculate_mitosis_potential(self) -> float:
        """Calculate potential for system division/mitosis"""
        mitosis_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.MITOSIS]
        
        if not mitosis_nodes:
            return 0.0
        
        # Mitosis potential based on mitosis nodes and system complexity
        complexity = len(self.nodes) / 100.0  # Normalize by system size
        potential = len(mitosis_nodes) * 0.4 * complexity
        return min(1.0, potential)


class AnkhRepairEngine:
    """
    Implements the Ankh process for recursive repair
    """
    
    def __init__(self, parser: RecursionPlaneParser):
        self.parser = parser
        self.repair_log = []
    
    def perform_ankh_repair(self, damaged_loop: RecursiveLoop, bridge_node: SymbolicNode) -> bool:
        """
        Perform Ankh repair on an incised loop
        
        Args:
            damaged_loop: The incised loop to repair
            bridge_node: Nexus node to use as repair bridge
            
        Returns:
            True if repair was successful
        """
        self.repair_log.append(f"Starting Ankh repair on loop at {damaged_loop.nodes[0].x},{damaged_loop.nodes[0].y}")
        
        # Phase 1: Identify damage points
        damage_points = self._identify_damage_points(damaged_loop)
        
        # Phase 2: Create dimensional bridge
        bridge_created = self._create_dimensional_bridge(damage_points, bridge_node)
        
        if bridge_created:
            # Phase 3: Redirect energy flow
            energy_redirected = self._redirect_energy_flow(damaged_loop, bridge_node)
            
            if energy_redirected:
                # Phase 4: Transform loop state
                transformation_success = self._transform_loop_state(damaged_loop)
                
                if transformation_success:
                    self.repair_log.append("Ankh repair completed successfully")
                    return True
                else:
                    self.repair_log.append("Loop state transformation failed")
            else:
                self.repair_log.append("Energy flow redirection failed")
        else:
            self.repair_log.append("Dimensional bridge creation failed")
        
        return False
    
    def _identify_damage_points(self, loop: RecursiveLoop) -> List[SymbolicNode]:
        """Identify points of damage in an incised loop"""
        damage_points = []
        
        for node in loop.nodes:
            # Damage points are where energy is leaking
            if len(node.connections) < 2:
                damage_points.append(node)
            elif node.node_type == NodeType.INCISED_LOOP:
                damage_points.append(node)
        
        return damage_points
    
    def _create_dimensional_bridge(self, damage_points: List[SymbolicNode], 
                                 bridge_node: SymbolicNode) -> bool:
        """Create a dimensional bridge across damaged points"""
        if not damage_points or not bridge_node:
            return False
        
        # Connect the bridge node to the damage points
        for damage_point in damage_points:
            if bridge_node not in damage_point.connections:
                damage_point.connections.append(bridge_node)
            if damage_point not in bridge_node.connections:
                bridge_node.connections.append(damage_point)
        
        self.repair_log.append(f"Dimensional bridge created at {bridge_node.x},{bridge_node.y}")
        return True
    
    def _redirect_energy_flow(self, loop: RecursiveLoop, bridge_node: SymbolicNode) -> bool:
        """Redirect energy flow through the dimensional bridge"""
        # Increase energy level of bridge node
        bridge_node.energy_level *= 1.5
        
        # Update connections to prefer the bridge
        for node in loop.nodes:
            if bridge_node in node.connections:
                # Prioritize the bridge connection
                node.connections = [bridge_node] + [c for c in node.connections if c != bridge_node]
        
        self.repair_log.append("Energy flow redirected through dimensional bridge")
        return True
    
    def _transform_loop_state(self, loop: RecursiveLoop) -> bool:
        """Transform incised loop to healed loop"""
        # Update the loop nodes
        for node in loop.nodes:
            if node.node_type == NodeType.INCISED_LOOP:
                node.node_type = NodeType.HEALED_LOOP
                node.symbol = "10"
        
        # Update loop metadata
        loop.loop_type = NodeType.HEALED_LOOP
        loop.stability = min(1.0, loop.stability * 1.8)  # Increase stability
        loop.energy_flow *= 1.2  # Increase energy flow
        loop.dimensional_integrity = True
        
        self.repair_log.append("Loop state transformed from incised to healed")
        return True


# Example usage and demonstration
if __name__ == "__main__":
    # Example diagram from the conversation
    example_diagram = """
    > x <
    _ x _
    9 x 10
    . x :
    """
    
    parser = RecursionPlaneParser()
    analysis = parser.parse_diagram(example_diagram)
    
    print("=== Recursion Plane Analysis ===")
    print(f"Total nodes: {analysis['total_nodes']}")
    print(f"Node distribution: {analysis['node_distribution']}")
    print(f"Total loops: {analysis['total_loops']}")
    print(f"Loop states: {analysis['loop_states']}")
    print(f"Crossover points: {analysis['total_crossovers']}")
    print(f"Ankh opportunities: {len(analysis['ankh_repair_opportunities'])}")
    
    # Demonstrate Ankh repair
    if analysis['ankh_repair_opportunities']:
        print("\n=== Ankh Repair Demonstration ===")
        repair_engine = AnkhRepairEngine(parser)
        
        # Find an incised loop and a nexus
        incised_loop = None
        nexus_node = None
        
        for loop in parser.loops:
            if loop.loop_type == NodeType.INCISED_LOOP:
                incised_loop = loop
                break
        
        for node in parser.nodes.values():
            if node.node_type == NodeType.NEXUS:
                nexus_node = node
                break
        
        if incised_loop and nexus_node:
            success = repair_engine.perform_ankh_repair(incised_loop, nexus_node)
            print(f"Repair successful: {success}")
            print("Repair log:")
            for log_entry in repair_engine.repair_log:
                print(f"  - {log_entry}")
