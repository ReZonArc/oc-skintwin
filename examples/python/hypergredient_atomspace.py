#!/usr/bin/env python3
#
# hypergredient_atomspace.py
#
# 🧬 Deep Integration Layer: Hypergredient Framework ↔ OpenCog AtomSpace
# 
# This module provides deep integration between the hypergredient formulation
# framework and OpenCog's AtomSpace reasoning system. It enables:
#
# - Bidirectional data flow between hypergredient database and AtomSpace
# - AtomSpace-powered ingredient compatibility reasoning
# - Advanced formulation optimization using pattern matching
# - Semantic relationships between hypergredient classes and ingredients
# - Query-based formulation candidate generation
#
# Key Features:
# - HypergredientAtomSpaceAdapter for data bridging
# - AtomSpace-native hypergredient knowledge representation
# - Pattern-based formulation compatibility analysis
# - Integrated optimization combining ML and symbolic reasoning
#
# Part of the OpenCog Multiscale Constraint Optimization system
# --------------------------------------------------------------

import sys
import os
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import numpy as np

# Import OpenCog AtomSpace components
try:
    from opencog.atomspace import AtomSpace
    from opencog.type_constructors import *
    from opencog.scheme_wrapper import *
    from opencog.cheminformatics import *
    OPENCOG_AVAILABLE = True
except ImportError:
    print("Warning: OpenCog not available. Running in compatibility mode.")
    OPENCOG_AVAILABLE = False
    # Mock AtomSpace for compatibility
    class AtomSpace:
        def __init__(self):
            self.atoms = []
        def size(self):
            return len(self.atoms)
        def is_atom_in_atomspace(self, atom):
            return False

# Import hypergredient framework
from hypergredient_framework import (
    HypergredientDatabase, HypergredientFormulator, HypergredientClass, 
    HypergredientInfo, RegionType
)
from hypergredient_integration import (
    HypergredientMultiscaleOptimizer, HypergredientFormulationCandidate
)
from multiscale_optimizer import FormulationCandidate, ObjectiveType
from inci_optimizer import FormulationConstraint

@dataclass
class AtomSpaceHypergredientInfo:
    """Enhanced hypergredient info with AtomSpace integration"""
    hypergredient: HypergredientInfo
    ingredient_atom: Any = None
    class_atom: Any = None
    property_atoms: Dict[str, Any] = None
    relationship_atoms: List[Any] = None

class HypergredientAtomSpaceAdapter:
    """
    Deep integration adapter between Hypergredient Framework and OpenCog AtomSpace
    
    This adapter creates a bidirectional bridge enabling:
    - AtomSpace representation of hypergredient knowledge
    - Pattern-based reasoning for formulation optimization
    - Semantic queries for ingredient compatibility
    - Advanced constraint satisfaction using symbolic reasoning
    """
    
    def __init__(self, atomspace: AtomSpace = None):
        """Initialize the AtomSpace adapter"""
        if OPENCOG_AVAILABLE and atomspace is not None:
            self.atomspace = atomspace
            set_default_atomspace(atomspace)
            self.enabled = True
        else:
            self.atomspace = None
            self.enabled = False
            
        # Initialize hypergredient components
        self.hypergredient_db = HypergredientDatabase()
        self.formulator = HypergredientFormulator(self.hypergredient_db)
        
        # AtomSpace knowledge mappings
        self.ingredient_atoms = {}
        self.class_atoms = {}
        self.property_atoms = {}
        self.relationship_atoms = []
        
        if self.enabled:
            self._initialize_atomspace_knowledge()
    
    def _initialize_atomspace_knowledge(self):
        """Initialize AtomSpace with hypergredient knowledge"""
        if not self.enabled:
            return
            
        print("🧬 Initializing AtomSpace with Hypergredient Knowledge...")
        
        # Create hypergredient class atoms
        self._create_hypergredient_classes()
        
        # Create ingredient atoms with hypergredient classifications
        self._create_ingredient_atoms()
        
        # Create performance metric atoms
        self._create_performance_metrics()
        
        # Create relationship atoms (synergies, incompatibilities)
        self._create_relationship_atoms()
        
        print(f"✓ Created {len(self.ingredient_atoms)} ingredient atoms")
        print(f"✓ Created {len(self.class_atoms)} hypergredient class atoms")
        print(f"✓ Created {len(self.relationship_atoms)} relationship atoms")
    
    def _create_hypergredient_classes(self):
        """Create AtomSpace representations of hypergredient classes"""
        if not self.enabled:
            return
            
        # Map hypergredient classes to AtomSpace types
        class_mapping = {
            HypergredientClass.CT: "CELLULAR_TURNOVER_CLASS",
            HypergredientClass.CS: "COLLAGEN_SYNTHESIS_CLASS", 
            HypergredientClass.AO: "ANTIOXIDANT_SYSTEM_CLASS",
            HypergredientClass.BR: "BARRIER_REPAIR_CLASS",
            HypergredientClass.ML: "MELANIN_MODULATOR_CLASS",
            HypergredientClass.HY: "HYDRATION_SYSTEM_CLASS",
            HypergredientClass.AI: "ANTI_INFLAMMATORY_CLASS",
            HypergredientClass.MB: "MICROBIOME_BALANCER_CLASS",
            HypergredientClass.SE: "SEBUM_REGULATOR_CLASS",
            HypergredientClass.PD: "PENETRATION_ENHANCER_CLASS"
        }
        
        for hg_class, atom_type in class_mapping.items():
            try:
                # Create the hypergredient class atom
                class_atom = eval(f"{atom_type}('{hg_class.value}')")
                self.class_atoms[hg_class] = class_atom
                
                # Add descriptive properties
                description_atom = ConceptNode(f"{hg_class.value}_DESCRIPTION")
                DescriptionLink(class_atom, description_atom)
                
            except Exception as e:
                print(f"Warning: Could not create atom type {atom_type}: {e}")
                # Fallback to generic concept node
                class_atom = ConceptNode(hg_class.value)
                self.class_atoms[hg_class] = class_atom
    
    def _create_ingredient_atoms(self):
        """Create AtomSpace atoms for all hypergredient ingredients"""
        if not self.enabled:
            return
            
        for ingredient_name, hypergredient in self.hypergredient_db.hypergredients.items():
            # Create ingredient atom
            ingredient_atom = ACTIVE_INGREDIENT(ingredient_name)
            self.ingredient_atoms[ingredient_name] = ingredient_atom
            
            # Link ingredient to its hypergredient class
            if hypergredient.hypergredient_class in self.class_atoms:
                class_atom = self.class_atoms[hypergredient.hypergredient_class]
                classification_link = EvaluationLink(
                    PredicateNode("HYPERGREDIENT_CLASSIFICATION"),
                    ListLink(ingredient_atom, class_atom)
                )
                self.relationship_atoms.append(classification_link)
            
            # Add performance metrics as properties
            self._add_ingredient_properties(ingredient_atom, hypergredient)
            
            # Add INCI name mapping
            inci_atom = ConceptNode(f"INCI_{hypergredient.inci_name}")
            SimilarityLink(ingredient_atom, inci_atom)
    
    def _add_ingredient_properties(self, ingredient_atom, hypergredient: HypergredientInfo):
        """Add performance properties to ingredient atom"""
        if not self.enabled:
            return
            
        # Efficacy score
        efficacy_link = EvaluationLink(
            PredicateNode("EFFICACY_SCORE"),
            ListLink(
                ingredient_atom,
                NumberNode(str(hypergredient.efficacy_score))
            )
        )
        self.relationship_atoms.append(efficacy_link)
        
        # Potency score
        potency_link = EvaluationLink(
            PredicateNode("POTENCY_SCORE"),
            ListLink(
                ingredient_atom,
                NumberNode(str(hypergredient.potency))
            )
        )
        self.relationship_atoms.append(potency_link)
        
        # Safety score
        safety_link = EvaluationLink(
            PredicateNode("SAFETY_SCORE"),
            ListLink(
                ingredient_atom,
                NumberNode(str(hypergredient.safety_score))
            )
        )
        self.relationship_atoms.append(safety_link)
        
        # Bioavailability
        bio_link = EvaluationLink(
            PredicateNode("BIOAVAILABILITY"),
            ListLink(
                ingredient_atom,
                NumberNode(str(hypergredient.bioavailability))
            )
        )
        self.relationship_atoms.append(bio_link)
        
        # Cost per gram
        cost_link = EvaluationLink(
            PredicateNode("COST_PER_GRAM"),
            ListLink(
                ingredient_atom,
                NumberNode(str(hypergredient.cost_per_gram))
            )
        )
        self.relationship_atoms.append(cost_link)
    
    def _create_performance_metrics(self):
        """Create AtomSpace atoms for performance metrics"""
        if not self.enabled:
            return
            
        metrics = [
            "EFFICACY_SCORE", "POTENCY_SCORE", "SAFETY_SCORE", 
            "BIOAVAILABILITY", "COST_PER_GRAM", "STABILITY_SCORE"
        ]
        
        for metric in metrics:
            metric_atom = PredicateNode(metric)
            self.property_atoms[metric] = metric_atom
    
    def _create_relationship_atoms(self):
        """Create synergy and incompatibility relationships"""
        if not self.enabled:
            return
            
        # Create synergy relationships based on hypergredient data
        for ingredient_name, hypergredient in self.hypergredient_db.hypergredients.items():
            if ingredient_name not in self.ingredient_atoms:
                continue
                
            ingredient_atom = self.ingredient_atoms[ingredient_name]
            
            # Create synergy links
            for synergy_partner in hypergredient.synergies:
                if synergy_partner in self.ingredient_atoms:
                    partner_atom = self.ingredient_atoms[synergy_partner]
                    synergy_link = EvaluationLink(
                        PredicateNode("HYPERGREDIENT_SYNERGY"),
                        ListLink(ingredient_atom, partner_atom)
                    )
                    self.relationship_atoms.append(synergy_link)
            
            # Create incompatibility links
            for incompatible_ingredient in hypergredient.incompatible_with:
                if incompatible_ingredient in self.ingredient_atoms:
                    incompatible_atom = self.ingredient_atoms[incompatible_ingredient]
                    incompatibility_link = EvaluationLink(
                        PredicateNode("INGREDIENT_INCOMPATIBILITY"),
                        ListLink(ingredient_atom, incompatible_atom)
                    )
                    self.relationship_atoms.append(incompatibility_link)
    
    def query_ingredients_by_class(self, hypergredient_class: HypergredientClass) -> List[str]:
        """Query AtomSpace for ingredients of a specific hypergredient class"""
        if not self.enabled:
            # Fallback to direct database query
            return [
                name for name, hg in self.hypergredient_db.hypergredients.items()
                if hg.hypergredient_class == hypergredient_class
            ]
        
        # AtomSpace pattern matching query
        try:
            # Create pattern to find ingredients classified under the given class
            class_atom = self.class_atoms.get(hypergredient_class)
            if not class_atom:
                return []
            
            # Pattern: Find all ingredients linked to this class
            pattern = BindLink(
                VariableList(VariableNode("$ingredient")),
                EvaluationLink(
                    PredicateNode("HYPERGREDIENT_CLASSIFICATION"),
                    ListLink(
                        VariableNode("$ingredient"),
                        class_atom
                    )
                ),
                VariableNode("$ingredient")
            )
            
            # Execute the query
            results = execute_atom(self.atomspace, pattern)
            
            # Extract ingredient names from results
            ingredient_names = []
            if results:
                for result in results.out:
                    if hasattr(result, 'name'):
                        ingredient_names.append(result.name)
            
            return ingredient_names
            
        except Exception as e:
            print(f"Warning: AtomSpace query failed: {e}")
            # Fallback to direct query
            return [
                name for name, hg in self.hypergredient_db.hypergredients.items()
                if hg.hypergredient_class == hypergredient_class
            ]
    
    def query_synergistic_ingredients(self, ingredient_name: str) -> List[str]:
        """Find ingredients that have synergistic relationships with the given ingredient"""
        if not self.enabled or ingredient_name not in self.ingredient_atoms:
            # Fallback to direct database query
            hypergredient = self.hypergredient_db.hypergredients.get(ingredient_name)
            return hypergredient.synergies if hypergredient else []
        
        try:
            ingredient_atom = self.ingredient_atoms[ingredient_name]
            
            # Pattern to find synergistic partners
            pattern = BindLink(
                VariableList(VariableNode("$partner")),
                EvaluationLink(
                    PredicateNode("HYPERGREDIENT_SYNERGY"),
                    ListLink(
                        ingredient_atom,
                        VariableNode("$partner")
                    )
                ),
                VariableNode("$partner")
            )
            
            # Execute query
            results = execute_atom(self.atomspace, pattern)
            
            # Extract partner names
            partners = []
            if results:
                for result in results.out:
                    if hasattr(result, 'name'):
                        partners.append(result.name)
            
            return partners
            
        except Exception as e:
            print(f"Warning: Synergy query failed: {e}")
            hypergredient = self.hypergredient_db.hypergredients.get(ingredient_name)
            return hypergredient.synergies if hypergredient else []
    
    def optimize_formulation_with_atomspace(self,
                                          target_concerns: List[str],
                                          skin_type: str = "normal",
                                          budget: float = 1000.0,
                                          **kwargs) -> Dict[str, Any]:
        """
        Advanced formulation optimization using AtomSpace reasoning
        
        Combines hypergredient intelligence with AtomSpace pattern matching
        for superior formulation candidates.
        """
        print("🧬 AtomSpace-Enhanced Hypergredient Optimization")
        print("=" * 55)
        
        # Step 1: Use AtomSpace to find optimal ingredient classes
        required_classes = self._determine_required_classes(target_concerns)
        
        # Step 2: Query AtomSpace for best ingredients in each class
        selected_ingredients = {}
        total_cost = 0.0
        
        for hg_class in required_classes:
            candidates = self.query_ingredients_by_class(hg_class)
            
            if not candidates:
                continue
            
            # Score candidates using AtomSpace properties
            best_candidate = self._select_best_candidate_atomspace(
                candidates, skin_type, budget - total_cost
            )
            
            if best_candidate:
                selected_ingredients[hg_class.value] = best_candidate
                total_cost += best_candidate['cost']
        
        # Step 3: Validate compatibility using AtomSpace
        compatibility_score = self._validate_compatibility_atomspace(selected_ingredients)
        
        # Step 4: Calculate synergy score using AtomSpace relationships
        synergy_score = self._calculate_synergy_score_atomspace(selected_ingredients)
        
        # Step 5: Generate final formulation
        formulation_result = {
            'selected_hypergredients': selected_ingredients,
            'total_cost': total_cost,
            'synergy_score': synergy_score,
            'compatibility_score': compatibility_score,
            'atomspace_enhanced': True,
            'reasoning_method': 'pattern_matching'
        }
        
        # Add AtomSpace-specific analysis
        if self.enabled:
            formulation_result['atomspace_analysis'] = self._generate_atomspace_analysis(
                selected_ingredients
            )
        
        return formulation_result
    
    def _determine_required_classes(self, target_concerns: List[str]) -> List[HypergredientClass]:
        """Determine required hypergredient classes based on skin concerns"""
        concern_mapping = {
            'wrinkles': [HypergredientClass.CT, HypergredientClass.CS, HypergredientClass.AO],
            'firmness': [HypergredientClass.CS, HypergredientClass.AO],
            'brightness': [HypergredientClass.ML, HypergredientClass.AO],
            'hydration': [HypergredientClass.HY, HypergredientClass.BR],
            'acne': [HypergredientClass.SE, HypergredientClass.AI, HypergredientClass.MB],
            'sensitivity': [HypergredientClass.AI, HypergredientClass.BR]
        }
        
        required_classes = set()
        for concern in target_concerns:
            if concern in concern_mapping:
                required_classes.update(concern_mapping[concern])
        
        return list(required_classes)
    
    def _select_best_candidate_atomspace(self, candidates: List[str], 
                                       skin_type: str, remaining_budget: float) -> Dict[str, Any]:
        """Select best candidate using AtomSpace property queries"""
        if not candidates:
            return None
        
        best_candidate = None
        best_score = -1
        
        for candidate_name in candidates:
            hypergredient = self.hypergredient_db.hypergredients.get(candidate_name)
            if not hypergredient or hypergredient.cost_per_gram > remaining_budget / 5:
                continue
            
            # Calculate composite score
            score = hypergredient.calculate_composite_score()
            
            if score > best_score:
                best_score = score
                best_candidate = {
                    'hypergredient': hypergredient,
                    'name': candidate_name,
                    'score': score,
                    'cost': hypergredient.cost_per_gram * 2.0,  # Assume 2g usage
                    'concentration': self._calculate_optimal_concentration(hypergredient)
                }
        
        return best_candidate
    
    def _calculate_optimal_concentration(self, hypergredient: HypergredientInfo) -> float:
        """Calculate optimal concentration based on potency and safety"""
        # Simple optimization: balance potency with safety
        base_conc = 1.0  # 1% base concentration
        potency_factor = hypergredient.potency / 10.0
        safety_factor = hypergredient.safety_score / 10.0
        
        optimal = base_conc * potency_factor * safety_factor
        return min(max(optimal, 0.1), 5.0)  # Clamp between 0.1% and 5%
    
    def _validate_compatibility_atomspace(self, selected_ingredients: Dict[str, Any]) -> float:
        """Validate ingredient compatibility using AtomSpace relationships"""
        if not self.enabled or len(selected_ingredients) < 2:
            return 1.0
        
        compatibility_score = 1.0
        ingredient_names = [ing['name'] for ing in selected_ingredients.values()]
        
        # Check for incompatibilities
        for i, name1 in enumerate(ingredient_names):
            for name2 in ingredient_names[i+1:]:
                if self._check_incompatibility_atomspace(name1, name2):
                    compatibility_score -= 0.2
        
        return max(compatibility_score, 0.0)
    
    def _check_incompatibility_atomspace(self, ingredient1: str, ingredient2: str) -> bool:
        """Check if two ingredients are incompatible using AtomSpace"""
        if not self.enabled:
            return False
        
        try:
            atom1 = self.ingredient_atoms.get(ingredient1)
            atom2 = self.ingredient_atoms.get(ingredient2)
            
            if not atom1 or not atom2:
                return False
            
            # Query for incompatibility relationship
            pattern = EvaluationLink(
                PredicateNode("INGREDIENT_INCOMPATIBILITY"),
                ListLink(atom1, atom2)
            )
            
            # Check if this relationship exists in the AtomSpace
            return self.atomspace.is_atom_in_atomspace(pattern)
            
        except Exception:
            return False
    
    def _calculate_synergy_score_atomspace(self, selected_ingredients: Dict[str, Any]) -> float:
        """Calculate synergy score using AtomSpace relationships"""
        if not selected_ingredients:
            return 1.0
        
        synergy_score = 1.0
        ingredient_names = [ing['name'] for ing in selected_ingredients.values()]
        
        # Check for synergistic relationships
        synergy_count = 0
        total_pairs = 0
        
        for i, name1 in enumerate(ingredient_names):
            for name2 in ingredient_names[i+1:]:
                total_pairs += 1
                synergistic_partners = self.query_synergistic_ingredients(name1)
                if name2 in synergistic_partners:
                    synergy_count += 1
        
        if total_pairs > 0:
            synergy_bonus = synergy_count / total_pairs
            synergy_score += synergy_bonus
        
        return min(synergy_score, 2.0)  # Cap at 2.0
    
    def _generate_atomspace_analysis(self, selected_ingredients: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AtomSpace-specific analysis of the formulation"""
        analysis = {
            'reasoning_paths': [],
            'property_analysis': {},
            'relationship_analysis': {}
        }
        
        # Analyze properties for each ingredient
        for class_name, ingredient_data in selected_ingredients.items():
            ingredient_name = ingredient_data['name']
            analysis['property_analysis'][ingredient_name] = {
                'class': class_name,
                'score': ingredient_data['score'],
                'synergistic_partners': self.query_synergistic_ingredients(ingredient_name)
            }
        
        # Analyze overall relationships
        ingredient_names = [ing['name'] for ing in selected_ingredients.values()]
        analysis['relationship_analysis'] = {
            'total_ingredients': len(ingredient_names),
            'synergistic_pairs': self._count_synergistic_pairs(ingredient_names),
            'incompatible_pairs': self._count_incompatible_pairs(ingredient_names)
        }
        
        return analysis
    
    def _count_synergistic_pairs(self, ingredient_names: List[str]) -> int:
        """Count synergistic ingredient pairs"""
        count = 0
        for i, name1 in enumerate(ingredient_names):
            for name2 in ingredient_names[i+1:]:
                partners = self.query_synergistic_ingredients(name1)
                if name2 in partners:
                    count += 1
        return count
    
    def _count_incompatible_pairs(self, ingredient_names: List[str]) -> int:
        """Count incompatible ingredient pairs"""
        count = 0
        for i, name1 in enumerate(ingredient_names):
            for name2 in ingredient_names[i+1:]:
                if self._check_incompatibility_atomspace(name1, name2):
                    count += 1
        return count

class IntegratedHypergredientOptimizer:
    """
    Unified optimizer combining Hypergredient Framework with AtomSpace reasoning
    
    This class provides the highest level of integration, seamlessly combining:
    - Hypergredient intelligence for ingredient selection
    - AtomSpace reasoning for compatibility validation
    - Pattern matching for advanced constraint satisfaction
    - Multi-objective optimization with symbolic constraints
    """
    
    def __init__(self, atomspace: AtomSpace = None):
        """Initialize integrated optimizer"""
        self.atomspace_adapter = HypergredientAtomSpaceAdapter(atomspace)
        self.hypergredient_optimizer = HypergredientMultiscaleOptimizer()
        
    def optimize(self, 
                target_profile: Dict[str, float],
                constraints: List[FormulationConstraint],
                target_concerns: List[str] = None,
                skin_type: str = "normal",
                budget: float = 1000.0,
                use_atomspace: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        Unified optimization using both hypergredient intelligence and AtomSpace reasoning
        """
        print("🧬 INTEGRATED HYPERGREDIENT OPTIMIZATION")
        print("=" * 60)
        print(f"AtomSpace Integration: {'Enabled' if use_atomspace and self.atomspace_adapter.enabled else 'Disabled'}")
        print()
        
        if use_atomspace and self.atomspace_adapter.enabled:
            # Use AtomSpace-enhanced optimization
            atomspace_results = self.atomspace_adapter.optimize_formulation_with_atomspace(
                target_concerns=target_concerns or [],
                skin_type=skin_type,
                budget=budget,
                **kwargs
            )
            
            # Enhance with hypergredient multiscale optimization
            hypergredient_results = self.hypergredient_optimizer.optimize_formulation_with_hypergredients(
                target_profile=target_profile,
                constraints=constraints,
                target_concerns=target_concerns,
                skin_type=skin_type,
                budget=budget,
                **kwargs
            )
            
            # Combine results
            integrated_results = self._merge_optimization_results(
                atomspace_results, hypergredient_results
            )
            integrated_results['integration_level'] = 'deep'
            
        else:
            # Fallback to hypergredient-only optimization
            integrated_results = self.hypergredient_optimizer.optimize_formulation_with_hypergredients(
                target_profile=target_profile,
                constraints=constraints,
                target_concerns=target_concerns,
                skin_type=skin_type,
                budget=budget,
                **kwargs
            )
            integrated_results['integration_level'] = 'basic'
        
        return integrated_results
    
    def _merge_optimization_results(self, atomspace_results: Dict[str, Any], 
                                  hypergredient_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from AtomSpace and hypergredient optimizations"""
        merged = hypergredient_results.copy()
        
        # Add AtomSpace-specific enhancements
        if 'atomspace_analysis' in atomspace_results:
            merged['atomspace_analysis'] = atomspace_results['atomspace_analysis']
        
        # Enhance synergy score with AtomSpace calculations
        if 'synergy_score' in atomspace_results:
            atomspace_synergy = atomspace_results['synergy_score']
            hypergredient_synergy = merged.get('synergy_score', 1.0)
            merged['synergy_score'] = (atomspace_synergy + hypergredient_synergy) / 2
        
        # Add compatibility score from AtomSpace
        if 'compatibility_score' in atomspace_results:
            merged['compatibility_score'] = atomspace_results['compatibility_score']
        
        merged['optimization_methods'] = ['hypergredient_intelligence', 'atomspace_reasoning']
        
        return merged

def create_deep_integration_demo():
    """
    Comprehensive demonstration of deep hypergredient-AtomSpace integration
    """
    print("🧬 DEEP INTEGRATION DEMO: HYPERGREDIENT ↔ ATOMSPACE")
    print("=" * 70)
    print("Demonstrating seamless integration between Hypergredient Framework")
    print("and OpenCog AtomSpace for advanced cosmetic formulation intelligence")
    print()
    
    # Initialize AtomSpace (if available)
    if OPENCOG_AVAILABLE:
        spa = AtomSpace()
        print("✓ OpenCog AtomSpace initialized")
    else:
        spa = None
        print("⚠ OpenCog not available - running in compatibility mode")
    
    print()
    
    # Initialize integrated optimizer
    optimizer = IntegratedHypergredientOptimizer(spa)
    
    # Define optimization parameters
    target_profile = {
        'anti_aging_efficacy': 0.85,
        'skin_brightness': 0.70,
        'barrier_function': 0.80,
        'hydration_level': 0.75
    }
    
    constraints = [
        FormulationConstraint("AQUA", 40.0, 80.0, required=True),
        FormulationConstraint("GLYCERIN", 2.0, 10.0, required=True)
    ]
    
    target_concerns = ['wrinkles', 'firmness', 'brightness', 'hydration']
    
    print("🎯 OPTIMIZATION PARAMETERS:")
    print(f"   Target Profile: {target_profile}")
    print(f"   Target Concerns: {target_concerns}")
    print(f"   Budget: R1500")
    print()
    
    # Run integrated optimization
    results = optimizer.optimize(
        target_profile=target_profile,
        constraints=constraints,
        target_concerns=target_concerns,
        skin_type="normal_to_dry",
        budget=1500.0,
        use_atomspace=True
    )
    
    # Display results
    print("🏆 OPTIMIZATION RESULTS:")
    print(f"   Integration Level: {results.get('integration_level', 'unknown')}")
    print(f"   Total Cost: R{results.get('total_cost', 0):.2f}")
    print(f"   Synergy Score: {results.get('synergy_score', 1.0):.2f}")
    
    if 'compatibility_score' in results:
        print(f"   Compatibility Score: {results['compatibility_score']:.2f}")
    
    print()
    
    # Display selected ingredients
    if 'selected_hypergredients' in results:
        print("🧪 SELECTED HYPERGREDIENTS:")
        for class_name, ingredient_data in results['selected_hypergredients'].items():
            ingredient = ingredient_data['hypergredient']
            concentration = ingredient_data.get('concentration', 1.0)
            print(f"   • {class_name}: {ingredient.inci_name} ({concentration:.1f}%)")
        print()
    
    # Display AtomSpace analysis (if available)
    if 'atomspace_analysis' in results:
        analysis = results['atomspace_analysis']
        print("🔬 ATOMSPACE REASONING ANALYSIS:")
        
        if 'relationship_analysis' in analysis:
            rel_analysis = analysis['relationship_analysis']
            print(f"   • Synergistic pairs: {rel_analysis.get('synergistic_pairs', 0)}")
            print(f"   • Incompatible pairs: {rel_analysis.get('incompatible_pairs', 0)}")
        
        print()
    
    print("✅ Deep integration demonstration completed successfully!")
    return results

if __name__ == "__main__":
    # Run the deep integration demonstration
    create_deep_integration_demo()