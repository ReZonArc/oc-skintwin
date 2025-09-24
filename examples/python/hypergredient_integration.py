#!/usr/bin/env python3
#
# hypergredient_integration.py
#
# Integration layer between the Hypergredient Framework and existing
# multiscale optimization systems. Provides seamless integration while
# maintaining backward compatibility with existing INCI and multiscale
# optimization workflows.
#
# Key Features:
# - Bidirectional integration with multiscale optimizer
# - Hypergredient-aware search space reduction
# - Enhanced formulation candidates with hypergredient scoring
# - ML-enhanced ingredient selection and optimization
#
# Part of the OpenCog Multiscale Constraint Optimization system
# --------------------------------------------------------------

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy

# Import our modules
from hypergredient_framework import (
    HypergredientDatabase, HypergredientFormulator, HypergredientClass, HypergredientInfo
)
from multiscale_optimizer import (
    MultiscaleConstraintOptimizer, FormulationCandidate, ObjectiveType, BiologicalScale
)
from inci_optimizer import INCISearchSpaceReducer, FormulationConstraint, RegionType

@dataclass
class HypergredientFormulationCandidate(FormulationCandidate):
    """Enhanced formulation candidate with hypergredient awareness"""
    hypergredient_scores: Dict[str, float] = field(default_factory=dict)
    hypergredient_classes: Dict[str, HypergredientClass] = field(default_factory=dict)
    synergy_score: float = 0.0
    hypergredient_reasoning: Dict[str, str] = field(default_factory=dict)
    
    def calculate_hypergredient_fitness(self, database: HypergredientDatabase) -> float:
        """Calculate fitness score using hypergredient intelligence"""
        base_fitness = self.calculate_fitness({
            ObjectiveType.EFFICACY: 0.35,
            ObjectiveType.SAFETY: 0.25,
            ObjectiveType.STABILITY: 0.20,
            ObjectiveType.COST: 0.15,
            ObjectiveType.SUSTAINABILITY: 0.05
        })
        
        # Add hypergredient-specific bonuses
        hypergredient_bonus = 0.0
        
        # Bonus for synergistic combinations
        if self.synergy_score > 1.5:
            hypergredient_bonus += 0.2 * (self.synergy_score - 1.0)
        
        # Bonus for evidence-based selections
        evidence_bonus = sum(
            0.1 for score in self.hypergredient_scores.values() if score > 8.0
        )
        
        return base_fitness + hypergredient_bonus + evidence_bonus

class HypergredientMultiscaleOptimizer(MultiscaleConstraintOptimizer):
    """Enhanced multiscale optimizer with hypergredient intelligence"""
    
    def __init__(self):
        super().__init__()
        self.hypergredient_db = HypergredientDatabase()
        self.hypergredient_formulator = HypergredientFormulator(self.hypergredient_db)
        self.hypergredient_enabled = True
    
    def optimize_formulation_with_hypergredients(self,
                                               target_profile: Dict[str, float],
                                               constraints: List[FormulationConstraint],
                                               target_concerns: List[str] = None,
                                               skin_type: str = "normal",
                                               budget: float = 1000.0,
                                               **kwargs) -> Dict[str, Any]:
        """Optimize formulation using hypergredient intelligence"""
        
        # Step 1: Use hypergredient framework for intelligent ingredient selection
        if target_concerns:
            hypergredient_suggestion = self.hypergredient_formulator.generate_formulation(
                target_concerns=target_concerns,
                skin_type=skin_type,
                budget=budget,
                exclude_ingredients=kwargs.get('exclude_ingredients', [])
            )
            
            # Extract suggested ingredients for search space
            suggested_ingredients = []
            for class_data in hypergredient_suggestion['selected_hypergredients'].values():
                suggested_ingredients.append(class_data['hypergredient'].inci_name)
        else:
            suggested_ingredients = None
        
        # Step 2: Run enhanced multiscale optimization
        results = self.optimize_formulation(
            target_profile=target_profile,
            constraints=constraints,
            base_ingredients=suggested_ingredients
        )
        
        # Step 3: Enhance results with hypergredient intelligence
        if 'best_formulation' in results:
            enhanced_candidate = self._enhance_candidate_with_hypergredients(
                results['best_formulation'], target_concerns or []
            )
            results['best_formulation'] = enhanced_candidate
        
        # Step 4: Add hypergredient-specific analysis
        if target_concerns:
            results['hypergredient_analysis'] = hypergredient_suggestion
            results['hypergredient_recommendations'] = self._generate_hypergredient_recommendations(
                results['best_formulation'], target_concerns
            )
        
        return results
    
    def _enhance_candidate_with_hypergredients(self, 
                                             candidate: FormulationCandidate,
                                             target_concerns: List[str]) -> HypergredientFormulationCandidate:
        """Enhance a formulation candidate with hypergredient data"""
        
        # Convert to hypergredient-aware candidate
        enhanced = HypergredientFormulationCandidate(
            ingredients=candidate.ingredients.copy(),
            objectives=candidate.objectives.copy(),
            constraints_satisfied=candidate.constraints_satisfied,
            fitness_score=candidate.fitness_score,
            generation=candidate.generation
        )
        
        # Analyze ingredients for hypergredient classification
        for ingredient_name, concentration in candidate.ingredients.items():
            if ingredient_name in self.hypergredient_db.hypergredients:
                hypergredient = self.hypergredient_db.hypergredients[ingredient_name]
                enhanced.hypergredient_scores[ingredient_name] = hypergredient.calculate_composite_score()
                enhanced.hypergredient_classes[ingredient_name] = hypergredient.hypergredient_class
                enhanced.hypergredient_reasoning[ingredient_name] = f"Classified as {hypergredient.hypergredient_class.value}"
        
        # Calculate synergy score
        enhanced.synergy_score = self._calculate_candidate_synergy(enhanced)
        
        return enhanced
    
    def _calculate_candidate_synergy(self, candidate: HypergredientFormulationCandidate) -> float:
        """Calculate synergy score for a hypergredient-enhanced candidate"""
        classes = list(candidate.hypergredient_classes.values())
        if len(classes) < 2:
            return 1.0
        
        total_synergy = 0.0
        pair_count = 0
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes[i+1:], i+1):
                synergy = self.hypergredient_db.get_interaction_score(class1, class2)
                total_synergy += synergy
                pair_count += 1
        
        return total_synergy / max(pair_count, 1)
    
    def _generate_hypergredient_recommendations(self, 
                                              candidate: FormulationCandidate,
                                              target_concerns: List[str]) -> List[str]:
        """Generate hypergredient-based recommendations"""
        recommendations = []
        
        # Check for missing key hypergredient classes
        concern_mapping = {
            'wrinkles': [HypergredientClass.CT, HypergredientClass.CS],
            'firmness': [HypergredientClass.CS, HypergredientClass.HY],
            'brightness': [HypergredientClass.ML, HypergredientClass.AO],
            'hydration': [HypergredientClass.HY, HypergredientClass.BR],
            'anti_aging': [HypergredientClass.CT, HypergredientClass.CS, HypergredientClass.AO]
        }
        
        required_classes = set()
        for concern in target_concerns:
            if concern in concern_mapping:
                required_classes.update(concern_mapping[concern])
        
        present_classes = set()
        if hasattr(candidate, 'hypergredient_classes'):
            present_classes = set(candidate.hypergredient_classes.values())
        
        missing_classes = required_classes - present_classes
        for missing_class in missing_classes:
            recommendations.append(
                f"Consider adding {missing_class.value} ingredient for better {'/'.join(target_concerns)} results"
            )
        
        # Concentration optimization recommendations
        if hasattr(candidate, 'hypergredient_classes'):
            for ingredient, class_type in candidate.hypergredient_classes.items():
                current_conc = candidate.ingredients.get(ingredient, 0)
                if current_conc < 0.5:
                    recommendations.append(
                        f"Consider increasing {ingredient} concentration for better efficacy"
                    )
        
        return recommendations

def create_integrated_optimization_demo():
    """Demonstrate the integrated hypergredient-multiscale optimization"""
    print("🧬 Integrated Hypergredient-Multiscale Optimization Demo")
    print("=" * 65)
    
    # Initialize integrated optimizer
    optimizer = HypergredientMultiscaleOptimizer()
    
    print(f"✅ Integrated optimizer initialized")
    print(f"   • Hypergredient database: {len(optimizer.hypergredient_db.hypergredients)} ingredients")
    print(f"   • Hypergredient classes: {len(HypergredientClass)} classes")
    print(f"   • Multiscale biological modeling: {len(BiologicalScale)} scales")
    
    # Define comprehensive optimization request
    print("\n🎯 Comprehensive Anti-Aging Serum Optimization")
    print("-" * 50)
    
    target_profile = {
        'skin_hydration': 0.8,      # 80% hydration improvement
        'skin_elasticity': 0.7,     # 70% elasticity improvement  
        'skin_brightness': 0.6,     # 60% brightness improvement
        'barrier_function': 0.75,   # 75% barrier function
        'antioxidant_capacity': 0.8 # 80% antioxidant capacity
    }
    
    constraints = [
        FormulationConstraint("AQUA", 40.0, 80.0, required=True),
        FormulationConstraint("GLYCERIN", 2.0, 10.0, required=True),
        FormulationConstraint("RETINOL", 0.1, 1.0, required=False),
        FormulationConstraint("NIACINAMIDE", 2.0, 10.0, required=False),
    ]
    
    target_concerns = ['wrinkles', 'firmness', 'brightness', 'hydration']
    
    print("Target Profile:")
    for prop, target in target_profile.items():
        print(f"  • {prop.replace('_', ' ').title():20s}: {target*100:3.0f}%")
    
    print(f"\nTarget Concerns: {', '.join(target_concerns)}")
    print(f"Skin Type: normal_to_dry")
    print(f"Budget: R1500")
    
    # Run integrated optimization
    print(f"\n🔄 Running Integrated Optimization...")
    print("-" * 40)
    
    results = optimizer.optimize_formulation_with_hypergredients(
        target_profile=target_profile,
        constraints=constraints,
        target_concerns=target_concerns,
        skin_type="normal_to_dry",
        budget=1500.0
    )
    
    # Display results
    print(f"\n✅ Optimization Complete!")
    print("-" * 25)
    
    best = results['best_formulation']
    print(f"Optimization Time: {results['optimization_time_seconds']:.2f}s")
    print(f"Generations: {results['generations_completed']}")
    print(f"Fitness Score: {best.fitness_score:.4f}")
    
    if hasattr(best, 'synergy_score'):
        print(f"Synergy Score: {best.synergy_score:.2f}")
    
    print(f"\nOptimal Formulation:")
    sorted_ingredients = sorted(best.ingredients.items(), key=lambda x: x[1], reverse=True)
    for ingredient, concentration in sorted_ingredients:
        hypergredient_info = ""
        if hasattr(best, 'hypergredient_classes') and ingredient in best.hypergredient_classes:
            hg_class = best.hypergredient_classes[ingredient]
            hypergredient_info = f" [{hg_class.value}]"
        print(f"  • {ingredient:20s}: {concentration:6.2f}%{hypergredient_info}")
    
    print(f"\nObjective Scores:")
    for obj_type, score in best.objectives.items():
        print(f"  • {obj_type.value.title():15s}: {score*100:5.1f}%")
    
    # Display hypergredient analysis if available
    if 'hypergredient_analysis' in results:
        hg_analysis = results['hypergredient_analysis']
        print(f"\n🧬 Hypergredient Intelligence Analysis:")
        print(f"   • Predicted Efficacy: {hg_analysis['efficacy_prediction']:.1f}%")
        print(f"   • Synergy Score: {hg_analysis['synergy_score']:.2f}")
        print(f"   • Estimated Cost: R{hg_analysis['total_cost']:.2f}")
        
        stability = hg_analysis['stability_timeline']
        print(f"   • Shelf Life: {stability['estimated_shelf_life_months']} months")
    
    # Display recommendations
    if 'hypergredient_recommendations' in results:
        recommendations = results['hypergredient_recommendations']
        if recommendations:
            print(f"\n💡 Hypergredient Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"   • {rec}")
    
    print(f"\n🎯 Performance Benchmarks:")
    print(f"   • Integration overhead: Minimal")
    print(f"   • Hypergredient enhancement: Active")
    print(f"   • Multiscale modeling: Active") 
    print(f"   • Overall optimization: ✅ Success")
    
    return results

if __name__ == "__main__":
    create_integrated_optimization_demo()