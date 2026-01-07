"""
Interactive Recursion Plane Demonstration
Tests and demonstrates all concepts from the 2D recursion plane analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recursion_plane_parser import RecursionPlaneParser, AnkhRepairEngine, NodeType
from recursion_plane_visualizer import RecursionPlaneVisualizer
import matplotlib.pyplot as plt


class InteractiveRecursionDemo:
    """
    Interactive demonstration of recursion plane concepts
    """
    
    def __init__(self):
        self.parser = RecursionPlaneParser()
        self.visualizer = RecursionPlaneVisualizer()
        self.repair_engine = None
        
        # Predefined examples
        self.examples = {
            "basic_introduction": {
                "name": "Basic Introduction",
                "diagram": """
    > x <
    _ x _
    9 x 10
    . x :
    """,
                "description": "Basic symbols and loop states"
            },
            
            "growth_pattern": {
                "name": "Growth Pattern",
                "diagram": """
    > _ > _ >
    x _ x _ x
    < _ < _ <
    """,
                "description": "Sustained growth pathways"
            },
            
            "complex_network": {
                "name": "Complex Network",
                "diagram": """
    8 > x < 8
    _ x _ x _
    > x 9 x <
    _ x _ x _
    : x 10 x .
    """,
                "description": "Multiple loop states and crossovers"
            },
            
            "dimensional_crossover": {
                "name": "Dimensional Crossover",
                "diagram": """
    > x x <
    x 9 x 10
    < x x >
    """,
                "description": "Multiple crossover points"
            },
            
            "ankh_repair_ready": {
                "name": "Ankh Repair Ready",
                "diagram": """
    > x <
    x 9 x
    _ x _
    x 10x
    < x >
    """,
                "description": "System ready for Ankh repair"
            },
            
            "perfect_system": {
                "name": "Perfect System",
                "diagram": """
    8 x 8
    x _ x
    8 x 8
    """,
                "description": "Fully stable recursive system"
            }
        }
    
    def run_demo(self):
        """Run the interactive demonstration"""
        print("=== Interactive Recursion Plane Demonstration ===")
        print("Analyzing all possibilities and inferences from 2D recursion plane system")
        print()
        
        # Phase 1: Basic Analysis
        print("PHASE 1: BASIC SYMBOLIC ANALYSIS")
        print("=" * 50)
        self.demonstrate_basic_analysis()
        
        # Phase 2: Advanced Analysis
        print("\nPHASE 2: ADVANCED SYSTEM ANALYSIS")
        print("=" * 50)
        self.demonstrate_advanced_analysis()
        
        # Phase 3: Ankh Repair Demonstration
        print("\nPHASE 3: ANKH REPAIR PROCESS")
        print("=" * 50)
        self.demonstrate_ankh_repair()
        
        # Phase 4: Interactive Testing
        print("\nPHASE 4: INTERACTIVE TESTING")
        print("=" * 50)
        self.interactive_testing()
    
    def demonstrate_basic_analysis(self):
        """Demonstrate basic symbolic analysis"""
        for key, example in list(self.examples.items())[:3]:  # First 3 examples
            print(f"\nExample: {example['name']}")
            print(f"Description: {example['description']}")
            print(f"Diagram:\n{example['diagram']}")
            
            # Parse and analyze
            analysis = self.parser.parse_diagram(example['diagram'])
            
            print(f"Analysis:")
            print(f"  Total Nodes: {analysis['total_nodes']}")
            print(f"  Node Distribution: {analysis['node_distribution']}")
            print(f"  Total Loops: {analysis['total_loops']}")
            print(f"  Loop States: {analysis['loop_states']}")
            print(f"  Crossover Points: {analysis['total_crossovers']}")
            
            # Visualize
            fig = self.visualizer.visualize_static(example['diagram'], 
                                                 f"Analysis: {example['name']}")
            plt.show()
    
    def demonstrate_advanced_analysis(self):
        """Demonstrate advanced system analysis"""
        # Use complex network example
        example = self.examples["complex_network"]
        print(f"\nAdvanced Analysis: {example['name']}")
        print(f"Diagram:\n{example['diagram']}")
        
        analysis = self.parser.parse_diagram(example['diagram'])
        
        print("\nDetailed Analysis:")
        print(f"Dimensional Topology:")
        print(f"  Phase Count: {analysis['dimensional_topology']['phase_count']}")
        print(f"  Interdimensional Connections: {analysis['dimensional_topology']['interdimensional_connections']}")
        
        print(f"\nEnergy Distribution:")
        print(f"  Total Energy: {analysis['energy_distribution']['total_energy']:.2f}")
        print(f"  Average Energy: {analysis['energy_distribution']['average_energy']:.2f}")
        print(f"  Energy Concentration: {analysis['energy_distribution']['energy_concentration']:.3f}")
        
        print(f"\nStability Analysis:")
        print(f"  Overall Stability: {analysis['stability_analysis']['overall_stability']:.3f}")
        print(f"  Loop Stability: {analysis['stability_analysis']['loop_stability']:.3f}")
        print(f"  Connection Density: {analysis['stability_analysis']['connection_density']:.3f}")
        print(f"  Termination Balance: {analysis['stability_analysis']['termination_balance']:.3f}")
        
        print(f"\nTransformation Sequences:")
        for i, sequence in enumerate(analysis['transformation_sequences']):
            print(f"  Sequence {i+1}: {' → '.join(sequence)}")
        
        print(f"\nGrowth Potential: {analysis['growth_potential']:.3f}")
        print(f"Mitosis Potential: {analysis['mitosis_potential']:.3f}")
        
        # Visualize
        fig = self.visualizer.visualize_static(example['diagram'], 
                                             f"Advanced Analysis: {example['name']}")
        plt.show()
    
    def demonstrate_ankh_repair(self):
        """Demonstrate the Ankh repair process"""
        # Use Ankh repair ready example
        example = self.examples["ankh_repair_ready"]
        print(f"\nAnkh Repair Demonstration: {example['name']}")
        print(f"Diagram:\n{example['diagram']}")
        
        # Parse the diagram
        analysis = self.parser.parse_diagram(example['diagram'])
        self.repair_engine = AnkhRepairEngine(self.parser)
        
        # Find repair opportunities
        opportunities = analysis['ankh_repair_opportunities']
        print(f"\nRepair Opportunities Found: {len(opportunities)}")
        
        if opportunities:
            for i, opp in enumerate(opportunities):
                print(f"\nOpportunity {i+1}:")
                print(f"  Damaged Loop: {opp['damaged_loop']}")
                print(f"  Repair Bridge: {opp['repair_bridge']}")
                print(f"  Estimated Stability Gain: {opp['estimated_stability_gain']:.3f}")
                print(f"  Energy Flow Restoration: {opp['energy_flow_restoration']:.3f}")
                
                # Attempt repair
                print(f"\nAttempting Ankh Repair...")
                
                # Find the damaged loop and repair nexus
                damaged_loop = None
                for loop in self.parser.loops:
                    if (loop.nodes[0].x, loop.nodes[0].y) == tuple(opp['damaged_loop'][0]):
                        damaged_loop = loop
                        break
                
                repair_nexus = None
                for node in self.parser.nodes.values():
                    if (node.x, node.y) == tuple(opp['repair_bridge']):
                        repair_nexus = node
                        break
                
                if damaged_loop and repair_nexus:
                    success = self.repair_engine.perform_ankh_repair(damaged_loop, repair_nexus)
                    print(f"Repair {'SUCCESSFUL' if success else 'FAILED'}")
                    
                    if success:
                        print("\nRepair Log:")
                        for log_entry in self.repair_engine.repair_log:
                            print(f"  - {log_entry}")
                        
                        # Show before/after comparison
                        print("\nBEFORE REPAIR:")
                        fig1 = self.visualizer.visualize_static(example['diagram'], 
                                                              "BEFORE Ankh Repair")
                        plt.show()
                        
                        # Note: In a real implementation, we would need to regenerate
                        # the diagram with the updated state. For this demo, we'll
                        # just show the original again with a note.
                        print("\nAFTER REPAIR (simulated):")
                        print("The incised loop (9) has been transformed to healed loop (10)")
                        print("Energy flow has been restored through the dimensional bridge")
                        fig2 = self.visualizer.visualize_static(example['diagram'], 
                                                              "AFTER Ankh Repair (Simulated)")
                        plt.show()
                else:
                    print("Could not locate damaged loop or repair nexus")
        else:
            print("No repair opportunities found in this configuration")
    
    def interactive_testing(self):
        """Interactive testing with user input"""
        print("\nINTERACTIVE TESTING MODE")
        print("You can now test your own recursion plane diagrams!")
        print("\nSymbol Reference:")
        print("  > - Growth node    < - Mitosis node    x - Nexus junction")
        print("  . - Hard stop      : - Soft stop       8 - Perfect loop")
        print("  9 - Incised loop   10 - Healed loop   _ - Horizontal path")
        print("  \\ - Diagonal path  / - Vertical path")
        print("\nEnter your diagram (empty line to finish, 'quit' to exit):")
        
        while True:
            lines = []
            print("\nEnter diagram (one line at a time):")
            
            while True:
                line = input()
                if line.lower() == 'quit':
                    return
                if line.strip() == '':
                    break
                lines.append(line)
            
            if not lines:
                continue
            
            diagram = '\n'.join(lines)
            
            print(f"\nAnalyzing diagram:\n{diagram}")
            
            try:
                # Parse and analyze
                analysis = self.parser.parse_diagram(diagram)
                
                print("\nANALYSIS RESULTS:")
                print("=" * 40)
                print(f"Total Nodes: {analysis['total_nodes']}")
                print(f"Node Distribution: {analysis['node_distribution']}")
                print(f"Total Loops: {analysis['total_loops']}")
                print(f"Loop States: {analysis['loop_states']}")
                print(f"Crossover Points: {analysis['total_crossovers']}")
                print(f"Overall Stability: {analysis['stability_analysis']['overall_stability']:.3f}")
                print(f"Growth Potential: {analysis['growth_potential']:.3f}")
                print(f"Mitosis Potential: {analysis['mitosis_potential']:.3f}")
                print(f"Ankh Repair Opportunities: {len(analysis['ankh_repair_opportunities'])}")
                
                # Visualize
                fig = self.visualizer.visualize_static(diagram, "User Diagram Analysis")
                plt.show()
                
                # Detailed breakdown
                print("\nDetailed Analysis:")
                print(f"  Energy Distribution: {analysis['energy_distribution']['total_energy']:.2f} total")
                print(f"  Dimensional Phases: {analysis['dimensional_topology']['phase_count']}")
                print(f"  Transformation Sequences: {len(analysis['transformation_sequences'])}")
                
                if analysis['transformation_sequences']:
                    print("  Available Transformations:")
                    for seq in analysis['transformation_sequences']:
                        print(f"    {' → '.join(seq)}")
                
            except Exception as e:
                print(f"Error analyzing diagram: {e}")
                print("Please check your diagram format and try again.")
    
    def demonstrate_all_possibilities(self):
        """Demonstrate all possible states and transformations"""
        print("\nDEMONSTRATING ALL POSSIBILITIES")
        print("=" * 50)
        
        # Generate minimal examples of all key concepts
        examples = [
            ("Growth Sequence", "> _ > _ >", "Sequential growth expansion"),
            ("Mitosis Pattern", "< x <", "Division through nexus"),
            ("Perfect Loop", "8", "Self-sustaining recursion"),
            ("Incised Loop", "9", "Damaged recursion requiring repair"),
            ("Healed Loop", "10", "Repaired recursion with enhancement"),
            ("Dimensional Crossover", "> x <", "Growth-mitosis bridge"),
            ("Energy Flow", "> _ .", "Growth to termination"),
            ("Complex Network", "8 x 9 x 10", "Multiple loop states"),
            ("Stable System", "8 x 8", "Fully balanced recursion"),
            ("Repair Configuration", "9 x 10", "Ready for Ankh process")
        ]
        
        print("\nKey Concepts Demonstrated:")
        for i, (name, diagram, desc) in enumerate(examples, 1):
            print(f"\n{i}. {name}")
            print(f"   Diagram: {diagram}")
            print(f"   Description: {desc}")
            
            # Quick analysis
            analysis = self.parser.parse_diagram(diagram)
            print(f"   Nodes: {analysis['total_nodes']}, Loops: {analysis['total_loops']}")
            
            if analysis['total_crossovers'] > 0:
                print(f"   Crossovers: {analysis['total_crossovers']}")
            if len(analysis['ankh_repair_opportunities']) > 0:
                print(f"   Repair Opportunities: {len(analysis['ankh_repair_opportunities'])}")
        
        print(f"\nSUMMARY:")
        print(f"- {len(examples)} fundamental patterns demonstrated")
        print(f"- 8 primary node types analyzed")
        print(f"- 3 loop states characterized")
        print(f"- 1 repair process (Ankh) detailed")
        print(f"- Infinite possibilities from simple rules")


# Run the demonstration
if __name__ == "__main__":
    demo = InteractiveRecursionDemo()
    
    # Run main demo
    demo.run_demo()
    
    # Show all possibilities
    demo.demonstrate_all_possibilities()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("1. The 2D recursion plane provides a complete symbolic framework")
    print("2. Eight primary symbols encode dimensional relationships")
    print("3. The Ankh process enables systematic recursive repair")
    print("4. Energy distribution determines system stability")
    print("5. Dimensional crossovers create multi-phase computation")
    print("6. All possibilities emerge from simple grammatical rules")
    print("\nTranslation complete: All possibilities and inferences analyzed.")
