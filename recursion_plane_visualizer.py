"""
Recursion Plane Visualizer
Interactive demonstration of 2D recursion plane and Ankh repair process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Dict, List, Tuple, Optional
from recursion_plane_parser import RecursionPlaneParser, NodeType, RecursiveLoop, DimensionalCrossover, SymbolicNode


class RecursionPlaneVisualizer:
    """
    Visualizes the 2D recursion plane with dimensional encoding
    """
    
    def __init__(self, figsize=(12, 10)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.parser = RecursionPlaneParser()
        self.node_positions: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.node_artists: Dict[Tuple[int, int], patches.Circle] = {}
        self.connection_lines: List[plt.Line2D] = []
        self.animation_frames: List[np.ndarray] = []
        
        # Color scheme for different node types
        self.node_colors = {
            NodeType.GROWTH: '#00FF00',      # Bright green
            NodeType.MITOSIS: '#FF0000',     # Bright red  
            NodeType.NEXUS: '#FFFF00',       # Bright yellow
            NodeType.HARD_STOP: '#000000',   # Black
            NodeType.SOFT_STOP: '#808080',   # Gray
            NodeType.PERFECT_LOOP: '#00FFFF', # Cyan
            NodeType.INCISED_LOOP: '#FF8000', # Orange
            NodeType.HEALED_LOOP: '#8000FF',  # Purple
            NodeType.PATH_H: '#404040',      # Dark gray
            NodeType.PATH_D: '#404040',      # Dark gray
            NodeType.PATH_V: '#404040'       # Dark gray
        }
        
        # Symbol display mapping
        self.node_symbols = {
            NodeType.GROWTH: '>',
            NodeType.MITOSIS: '<',
            NodeType.NEXUS: 'x',
            NodeType.HARD_STOP: '.',
            NodeType.SOFT_STOP: ':',
            NodeType.PERFECT_LOOP: '8',
            NodeType.INCISED_LOOP: '9',
            NodeType.HEALED_LOOP: '10',
            NodeType.PATH_H: '_',
            NodeType.PATH_D: '\\',
            NodeType.PATH_V: '/'
        }
        
    def visualize_static(self, diagram: str, title: str = "Recursion Plane Analysis"):
        """
        Create a static visualization of the recursion plane
        
        Args:
            diagram: Symbolic diagram string
            title: Plot title
        """
        # Parse the diagram
        analysis = self.parser.parse_diagram(diagram)
        
        # Clear previous plot
        self.ax.clear()
        self.node_artists.clear()
        self.connection_lines.clear()
        
        # Set up the plot
        self.ax.set_title(title, fontsize=16, fontweight='bold', color='white')
        self.ax.set_facecolor('#1a1a1a')
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Calculate positions
        self._calculate_node_positions(diagram)
        
        # Draw connections first (so they appear behind nodes)
        self._draw_connections()
        
        # Draw nodes
        self._draw_nodes()
        
        # Draw dimensional crossovers
        self._draw_dimensional_crossovers()
        
        # Add annotations
        self._add_annotations(analysis)
        
        # Set axis properties
        self.ax.set_xlim(-1, max(pos[0] for pos in self.node_positions.values()) + 1)
        self.ax.set_ylim(-1, max(pos[1] for pos in self.node_positions.values()) + 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white')
        
        # Add legend
        self._add_legend()
        
        plt.tight_layout()
        return self.fig
    
    def visualize_ankh_repair(self, diagram: str, damaged_loop_pos: Tuple[int, int]):
        """
        Animate the Ankh repair process
        
        Args:
            diagram: Symbolic diagram string
            damaged_loop_pos: Position of damaged loop to repair
        """
        # Parse the diagram
        analysis = self.parser.parse_diagram(diagram)
        
        # Find the damaged loop
        damaged_loop = None
        for loop in self.parser.loops:
            if (loop.nodes[0].x, loop.nodes[0].y) == damaged_loop_pos:
                damaged_loop = loop
                break
        
        if not damaged_loop:
            print(f"No loop found at position {damaged_loop_pos}")
            return
        
        # Find a suitable nexus for repair
        repair_nexus = None
        for node in self.parser.nodes.values():
            if node.node_type == NodeType.NEXUS:
                # Check if nexus is near the damaged loop
                for loop_node in damaged_loop.nodes:
                    distance = abs(node.x - loop_node.x) + abs(node.y - loop_node.y)
                    if distance <= 2:
                        repair_nexus = node
                        break
                if repair_nexus:
                    break
        
        if not repair_nexus:
            print("No suitable nexus found for repair")
            return
        
        # Create animation
        self._create_repair_animation(damaged_loop, repair_nexus)
        
    def _calculate_node_positions(self, diagram: str):
        """Calculate screen positions for all nodes"""
        lines = diagram.strip().split('\n')
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char.strip() and char != ' ':
                    # Use a layout algorithm to position nodes
                    screen_x = x * 2  # Scale for better spacing
                    screen_y = -y * 2  # Negative for top-to-bottom layout
                    
                    self.node_positions[(x, y)] = (screen_x, screen_y)
    
    def _draw_connections(self):
        """Draw connections between nodes"""
        for pos, node in self.parser.nodes.items():
            if pos not in self.node_positions:
                continue
                
            start_x, start_y = self.node_positions[pos]
            
            for connection in node.connections:
                conn_pos = (connection.x, connection.y)
                if conn_pos in self.node_positions:
                    end_x, end_y = self.node_positions[conn_pos]
                    
                    # Determine connection type and style
                    if node.node_type == NodeType.GROWTH and connection.node_type == NodeType.MITOSIS:
                        # Dimensional crossover connection
                        line = plt.Line2D([start_x, end_x], [start_y, end_y], 
                                        color='#FF00FF', linewidth=3, alpha=0.7,
                                        linestyle='--', zorder=1)
                    elif node.node_type in [NodeType.PERFECT_LOOP, NodeType.HEALED_LOOP]:
                        # Loop connections
                        line = plt.Line2D([start_x, end_x], [start_y, end_y], 
                                        color='#00FFFF', linewidth=2, alpha=0.6,
                                        zorder=1)
                    else:
                        # Regular connections
                        line = plt.Line2D([start_x, end_x], [start_y, end_y], 
                                        color='white', linewidth=1, alpha=0.4,
                                        zorder=1)
                    
                    self.ax.add_line(line)
                    self.connection_lines.append(line)
    
    def _draw_nodes(self):
        """Draw all nodes with appropriate symbols and colors"""
        for pos, node in self.parser.nodes.items():
            if pos not in self.node_positions:
                continue
                
            screen_x, screen_y = self.node_positions[pos]
            
            # Create node circle
            circle = patches.Circle((screen_x, screen_y), radius=0.3, 
                                  facecolor=self.node_colors[node.node_type],
                                  edgecolor='white', linewidth=2, 
                                  alpha=0.9, zorder=3)
            
            self.ax.add_patch(circle)
            self.node_artists[pos] = circle
            
            # Add symbol text
            symbol_text = self.node_symbols[node.node_type]
            if node.node_type == NodeType.HEALED_LOOP:
                symbol_text = '10'
            
            self.ax.text(screen_x, screen_y, symbol_text, 
                        ha='center', va='center', fontsize=10, 
                        fontweight='bold', color='black', zorder=4)
    
    def _draw_dimensional_crossovers(self):
        """Highlight dimensional crossover points"""
        for crossover in self.parser.crossovers:
            x, y = crossover.point
            if (x, y) in self.node_positions:
                screen_x, screen_y = self.node_positions[(x, y)]
                
                # Add glow effect around crossover points
                glow = patches.Circle((screen_x, screen_y), radius=0.6,
                                    facecolor='none', edgecolor='#FF00FF',
                                    linewidth=4, alpha=0.5, zorder=2)
                self.ax.add_patch(glow)
                
                # Add energy transfer annotation
                self.ax.annotate(f'Crossover\\nEnergy: {crossover.energy_transfer:.2f}',
                               xy=(screen_x, screen_y), xytext=(screen_x + 1, screen_y + 1),
                               arrowprops=dict(arrowstyle='->', color='#FF00FF', lw=2),
                               fontsize=8, color='#FF00FF', ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', 
                                       edgecolor='#FF00FF', alpha=0.8))
    
    def _add_annotations(self, analysis: Dict[str, any]):
        """Add informational annotations to the plot"""
        # Create info panel
        info_text = f"""System Analysis:
        
Total Nodes: {analysis['total_nodes']}
Loop States: {analysis['loop_states']}
Crossover Points: {analysis['total_crossovers']}
Overall Stability: {analysis['stability_analysis']['overall_stability']:.2f}
Growth Potential: {analysis['growth_potential']:.2f}
Repair Opportunities: {len(analysis['ankh_repair_opportunities'])}"""
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=10, color='white', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', 
                            edgecolor='white', alpha=0.9))
    
    def _add_legend(self):
        """Add a legend explaining node types"""
        legend_elements = []
        
        for node_type, color in self.node_colors.items():
            if node_type in [NodeType.PATH_H, NodeType.PATH_D, NodeType.PATH_V]:
                continue  # Skip pathway markers in legend
                
            symbol = self.node_symbols[node_type]
            if node_type == NodeType.HEALED_LOOP:
                symbol = '10'
            
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='white', 
                            label=f'{symbol} - {node_type.name.replace("_", " ").title()}')
            )
        
        # Add dimensional crossover to legend
        legend_elements.append(
            patches.Patch(facecolor='none', edgecolor='#FF00FF', linewidth=3,
                        label='Dimensional Crossover')
        )
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      fontsize=9, facecolor='#2a2a2a', edgecolor='white',
                      labelcolor='white')
    
    def _create_repair_animation(self, damaged_loop: RecursiveLoop, repair_nexus: SymbolicNode):
        """Create animation of the Ankh repair process"""
        # Set up animation figure
        anim_fig, anim_ax = plt.subplots(figsize=(12, 10))
        anim_ax.set_facecolor('#1a1a1a')
        anim_fig.patch.set_facecolor('#0a0a0a')
        
        # Animation phases
        phases = [
            "Phase 1: Damage Detection",
            "Phase 2: Crossover Bridge Creation", 
            "Phase 3: Energy Flow Redirection",
            "Phase 4: Loop State Transformation"
        ]
        
        def animate(frame):
            anim_ax.clear()
            
            phase = frame // 30  # 30 frames per phase
            sub_frame = frame % 30
            
            if phase >= len(phases):
                phase = len(phases) - 1
                sub_frame = 29
            
            # Title
            anim_ax.set_title(f"Ankh Repair Process - {phases[phase]}", 
                            fontsize=16, fontweight='bold', color='white')
            
            # Draw base diagram
            self._draw_base_for_animation(anim_ax)
            
            # Add phase-specific effects
            if phase == 0:  # Damage Detection
                self._animate_damage_detection(anim_ax, damaged_loop, sub_frame)
            elif phase == 1:  # Bridge Creation
                self._animate_bridge_creation(anim_ax, repair_nexus, sub_frame)
            elif phase == 2:  # Energy Redirection
                self._animate_energy_redirection(anim_ax, damaged_loop, repair_nexus, sub_frame)
            elif phase == 3:  # State Transformation
                self._animate_state_transformation(anim_ax, damaged_loop, sub_frame)
            
            anim_ax.set_xlim(-2, 8)
            anim_ax.set_ylim(-6, 2)
            anim_ax.set_aspect('equal')
            anim_ax.grid(True, alpha=0.3, color='gray')
            anim_ax.tick_params(colors='white')
        
        # Create animation
        anim = FuncAnimation(anim_fig, animate, frames=120, interval=100, blit=False)
        plt.show()
        
        return anim
    
    def _draw_base_for_animation(self, ax):
        """Draw the base diagram for animation"""
        # Draw connections
        for line in self.connection_lines:
            ax.add_line(plt.Line2D(line.get_xdata(), line.get_ydata(),
                                 color=line.get_color(), linewidth=line.get_linewidth(),
                                 alpha=line.get_alpha(), zorder=line.get_zorder()))
        
        # Draw nodes
        for pos, circle in self.node_artists.items():
            ax.add_patch(patches.Circle(circle.center, circle.radius,
                                      facecolor=circle.get_facecolor(),
                                      edgecolor=circle.get_edgecolor(),
                                      linewidth=circle.get_linewidth(),
                                      alpha=circle.get_alpha(), zorder=circle.get_zorder()))
    
    def _animate_damage_detection(self, ax, damaged_loop: RecursiveLoop, frame):
        """Animate damage detection phase"""
        # Highlight damaged nodes with pulsing effect
        alpha = 0.5 + 0.5 * np.sin(frame * 0.3)
        
        for node in damaged_loop.nodes:
            if (node.x, node.y) in self.node_positions:
                screen_x, screen_y = self.node_positions[(node.x, node.y)]
                
                # Add damage indicator
                damage_circle = patches.Circle((screen_x, screen_y), radius=0.8,
                                             facecolor='none', edgecolor='#FF4444',
                                             linewidth=4, alpha=alpha, zorder=5)
                ax.add_patch(damage_circle)
                
                # Add energy leakage visualization
                if frame > 15:
                    for i in range(4):
                        angle = i * np.pi / 2 + frame * 0.1
                        leak_x = screen_x + 1.2 * np.cos(angle)
                        leak_y = screen_y + 1.2 * np.sin(angle)
                        ax.plot(leak_x, leak_y, 'ro', alpha=0.7, markersize=8, zorder=6)
    
    def _animate_bridge_creation(self, ax, repair_nexus: SymbolicNode, frame):
        """Animate dimensional bridge creation"""
        if (repair_nexus.x, repair_nexus.y) not in self.node_positions:
            return
        
        screen_x, screen_y = self.node_positions[(repair_nexus.x, repair_nexus.y)]
        
        # Growing bridge effect
        if frame < 15:
            radius = frame * 0.05
            bridge_circle = patches.Circle((screen_x, screen_y), radius=radius,
                                         facecolor='#FF00FF', edgecolor='white',
                                         linewidth=3, alpha=0.8, zorder=6)
            ax.add_patch(bridge_circle)
        else:
            # Pulsing bridge
            alpha = 0.6 + 0.4 * np.sin((frame - 15) * 0.4)
            bridge_circle = patches.Circle((screen_x, screen_y), radius=0.7,
                                         facecolor='#FF00FF', edgecolor='white',
                                         linewidth=3, alpha=alpha, zorder=6)
            ax.add_patch(bridge_circle)
            
            # Energy beams
            for i in range(8):
                angle = i * np.pi / 4
                beam_x = screen_x + 1.0 * np.cos(angle)
                beam_y = screen_y + 1.0 * np.sin(angle)
                ax.plot([screen_x, beam_x], [screen_y, beam_y], 
                       'm-', linewidth=3, alpha=0.7, zorder=5)
    
    def _animate_energy_redirection(self, ax, damaged_loop: RecursiveLoop, 
                                  repair_nexus: SymbolicNode, frame):
        """Animate energy flow redirection"""
        # Draw energy flow lines
        nexus_screen = self.node_positions.get((repair_nexus.x, repair_nexus.y))
        if not nexus_screen:
            return
        
        for node in damaged_loop.nodes:
            node_screen = self.node_positions.get((node.x, node.y))
            if node_screen:
                # Flowing energy particles
                t = (frame + hash((node.x, node.y)) % 30) / 30.0
                if t <= 1.0:
                    particle_x = nexus_screen[0] + t * (node_screen[0] - nexus_screen[0])
                    particle_y = nexus_screen[1] + t * (node_screen[1] - nexus_screen[1])
                    
                    ax.plot(particle_x, particle_y, 'yo', markersize=10, 
                           alpha=0.8, zorder=7)
    
    def _animate_state_transformation(self, ax, damaged_loop: RecursiveLoop, frame):
        """Animate loop state transformation"""
        if frame < 15:
            # Flickering transformation
            for node in damaged_loop.nodes:
                if (node.x, node.y) in self.node_positions:
                    screen_x, screen_y = self.node_positions[(node.x, node.y)]
                    
                    if frame % 2 == 0:
                        color = '#FF8000'  # Incised orange
                        symbol = '9'
                    else:
                        color = '#8000FF'  # Healed purple
                        symbol = '10'
                    
                    circle = patches.Circle((screen_x, screen_y), radius=0.3,
                                          facecolor=color, edgecolor='white',
                                          linewidth=2, alpha=0.9, zorder=8)
                    ax.add_patch(circle)
                    
                    ax.text(screen_x, screen_y, symbol, ha='center', va='center',
                           fontsize=10, fontweight='bold', color='black', zorder=9)
        else:
            # Final healed state
            for node in damaged_loop.nodes:
                if (node.x, node.y) in self.node_positions:
                    screen_x, screen_y = self.node_positions[(node.x, node.y)]
                    
                    circle = patches.Circle((screen_x, screen_y), radius=0.3,
                                          facecolor='#8000FF', edgecolor='white',
                                          linewidth=2, alpha=0.9, zorder=8)
                    ax.add_patch(circle)
                    
                    ax.text(screen_x, screen_y, '10', ha='center', va='center',
                           fontsize=10, fontweight='bold', color='black', zorder=9)
                    
                    # Healing glow
                    glow = patches.Circle((screen_x, screen_y), radius=0.6,
                                        facecolor='none', edgecolor='#8000FF',
                                        linewidth=3, alpha=0.6, zorder=7)
                    ax.add_patch(glow)


# Demonstration and example usage
if __name__ == "__main__":
    # Create example diagrams
    examples = {
        "Simple Recursion Plane": """
    > x <
    _ x _
    9 x 10
    . x :
    """,
        
        "Complex Network": """
    8 > x < 8
    _ x _ x _
    > x 9 x <
    _ x _ x _
    : x 10 x .
    """,
        
        "Growth Pattern": """
    > _ > _ >
    x _ x _ x
    < _ < _ <
    """,
        
        "Mixed States": """
    8 x 9 x 10
    _ x _ x _
    > x < x .
    """
    }
    
    # Create visualizer
    visualizer = RecursionPlaneVisualizer()
    
    # Visualize each example
    for name, diagram in examples.items():
        print(f"\nVisualizing: {name}")
        fig = visualizer.visualize_static(diagram, name)
        plt.show()
        
        # Print analysis
        analysis = visualizer.parser.parse_diagram(diagram)
        print(f"Analysis:")
        print(f"  - Total nodes: {analysis['total_nodes']}")
        print(f"  - Loop states: {analysis['loop_states']}")
        print(f"  - Crossover points: {analysis['total_crossovers']}")
        print(f"  - Ankh opportunities: {len(analysis['ankh_repair_opportunities'])}")
