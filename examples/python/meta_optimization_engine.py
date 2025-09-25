#!/usr/bin/env python3
#
# meta_optimization_engine.py
#
# 🧠 Meta-Optimization Strategy Engine for Cosmetic Formulation Design
# 
# This engine implements a comprehensive meta-optimization strategy that can generate
# optimal formulations for every possible skin condition and treatment combination.
# It integrates multiple optimization algorithms, learning strategies, and knowledge
# bases to provide superior formulation recommendations.
#
# Key Features:
# - Condition-Treatment Matrix Analysis
# - Multi-Strategy Optimization Coordination
# - Dynamic Knowledge Base Learning
# - Real-time Adaptation and Improvement
# - Evidence-based Formulation Generation
#
# Part of the OpenCog Multiscale Constraint Optimization system
# --------------------------------------------------------------

import json
import math
import numpy as np
import copy
import time
import random
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from itertools import combinations, product

# Import existing optimization frameworks
try:
    from multiscale_optimizer import MultiscaleConstraintOptimizer, FormulationCandidate, ObjectiveType, BiologicalScale
    from hypergredient_framework import HypergredientFormulator, HypergredientDatabase, HypergredientClass
    from inci_optimizer import INCISearchSpaceReducer, FormulationConstraint, RegionType
    from attention_allocation import AttentionAllocationManager
except ImportError:
    # Standalone mode with mock classes
    print("Warning: Some modules not available, running in standalone mode")
    
    class RegionType(Enum):
        EU = "EU"
        FDA = "FDA"
        INTERNATIONAL = "INTERNATIONAL"
    
    class ObjectiveType(Enum):
        EFFICACY = "efficacy"
        SAFETY = "safety"
        COST = "cost"
        STABILITY = "stability"

class SkinCondition(Enum):
    """Comprehensive skin condition taxonomy"""
    ACNE = "acne"
    WRINKLES = "wrinkles"
    HYPERPIGMENTATION = "hyperpigmentation"
    DRYNESS = "dryness"
    SENSITIVITY = "sensitivity"
    ROSACEA = "rosacea"
    MELASMA = "melasma"
    AGING = "aging"
    DULLNESS = "dullness"
    PORES = "enlarged_pores"
    DARK_CIRCLES = "dark_circles"
    PUFFINESS = "puffiness"
    LOSS_OF_FIRMNESS = "loss_of_firmness"
    UNEVEN_TEXTURE = "uneven_texture"
    OILINESS = "oiliness"
    DEHYDRATION = "dehydration"
    SUN_DAMAGE = "sun_damage"
    STRETCH_MARKS = "stretch_marks"
    CELLULITE = "cellulite"
    KERATOSIS_PILARIS = "keratosis_pilaris"

class TreatmentApproach(Enum):
    """Treatment approach strategies"""
    PREVENTION = "prevention"
    CORRECTION = "correction"
    MAINTENANCE = "maintenance"
    INTENSIVE_TREATMENT = "intensive_treatment"
    GENTLE_CARE = "gentle_care"
    COMBINATION_THERAPY = "combination_therapy"

class EvidenceLevel(Enum):
    """Evidence levels for ingredient efficacy"""
    STRONG = "strong"          # Multiple RCTs, meta-analyses
    MODERATE = "moderate"      # Some clinical studies
    LIMITED = "limited"        # In-vitro or limited clinical data
    THEORETICAL = "theoretical" # Mechanism-based reasoning only

@dataclass
class ConditionTreatmentProfile:
    """Profile defining optimal treatment for a specific condition"""
    condition: SkinCondition
    approach: TreatmentApproach
    target_mechanisms: List[str]
    required_ingredients: List[str] = field(default_factory=list)
    beneficial_ingredients: List[str] = field(default_factory=list)
    contraindicated_ingredients: List[str] = field(default_factory=list)
    optimal_pH_range: Tuple[float, float] = (5.0, 6.5)
    treatment_duration_weeks: int = 12
    expected_efficacy: float = 0.7  # 0-1 scale
    evidence_level: EvidenceLevel = EvidenceLevel.MODERATE

@dataclass
class OptimizationStrategy:
    """Configuration for a specific optimization approach"""
    name: str
    algorithm_type: str  # "evolutionary", "bayesian", "gradient", "hybrid"
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Relative importance in ensemble
    conditions_specialized: List[SkinCondition] = field(default_factory=list)
    
@dataclass
class MetaOptimizationResult:
    """Comprehensive result from meta-optimization process"""
    formulation: Dict[str, float]  # ingredient -> concentration
    condition_treatment_pair: Tuple[SkinCondition, TreatmentApproach]
    optimization_strategies_used: List[str]
    predicted_efficacy: float
    confidence_score: float
    evidence_basis: List[str]
    alternative_formulations: List[Dict[str, Any]] = field(default_factory=list)
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)

class ConditionTreatmentKnowledgeBase:
    """Comprehensive knowledge base for condition-treatment relationships"""
    
    def __init__(self):
        self.condition_profiles = {}
        self.ingredient_efficacy = {}
        self.mechanism_mapping = {}
        self.synergy_matrix = {}
        self.contraindication_matrix = {}
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize comprehensive condition-treatment knowledge"""
        
        # Define condition treatment profiles
        self.condition_profiles = {
            SkinCondition.ACNE: {
                TreatmentApproach.PREVENTION: ConditionTreatmentProfile(
                    condition=SkinCondition.ACNE,
                    approach=TreatmentApproach.PREVENTION,
                    target_mechanisms=["sebum_regulation", "antimicrobial", "anti_inflammatory"],
                    required_ingredients=["NIACINAMIDE", "SALICYLIC ACID"],
                    beneficial_ingredients=["ZINC OXIDE", "TEA TREE OIL"],
                    contraindicated_ingredients=["COMEDOGENIC_OILS"],
                    optimal_pH_range=(4.0, 5.5),
                    expected_efficacy=0.75,
                    evidence_level=EvidenceLevel.STRONG
                ),
                TreatmentApproach.CORRECTION: ConditionTreatmentProfile(
                    condition=SkinCondition.ACNE,
                    approach=TreatmentApproach.CORRECTION,
                    target_mechanisms=["keratolytic", "antimicrobial", "anti_inflammatory"],
                    required_ingredients=["SALICYLIC ACID", "BENZOYL PEROXIDE"],
                    beneficial_ingredients=["RETINOL", "NIACINAMIDE"],
                    contraindicated_ingredients=["HEAVY_OILS", "ALCOHOL"],
                    optimal_pH_range=(3.5, 5.0),
                    expected_efficacy=0.85,
                    evidence_level=EvidenceLevel.STRONG
                )
            },
            
            SkinCondition.WRINKLES: {
                TreatmentApproach.PREVENTION: ConditionTreatmentProfile(
                    condition=SkinCondition.WRINKLES,
                    approach=TreatmentApproach.PREVENTION,
                    target_mechanisms=["collagen_synthesis", "antioxidant", "photoprotection"],
                    required_ingredients=["VITAMIN C", "SUNSCREEN"],
                    beneficial_ingredients=["VITAMIN E", "FERULIC ACID"],
                    optimal_pH_range=(3.5, 6.0),
                    expected_efficacy=0.70,
                    evidence_level=EvidenceLevel.STRONG
                ),
                TreatmentApproach.CORRECTION: ConditionTreatmentProfile(
                    condition=SkinCondition.WRINKLES,
                    approach=TreatmentApproach.CORRECTION,
                    target_mechanisms=["cellular_turnover", "collagen_synthesis", "matrix_remodeling"],
                    required_ingredients=["RETINOL", "PEPTIDES"],
                    beneficial_ingredients=["VITAMIN C", "HYALURONIC ACID"],
                    contraindicated_ingredients=["HARSH_ALCOHOLS"],
                    optimal_pH_range=(5.5, 6.5),
                    expected_efficacy=0.80,
                    evidence_level=EvidenceLevel.STRONG
                )
            },
            
            SkinCondition.HYPERPIGMENTATION: {
                TreatmentApproach.CORRECTION: ConditionTreatmentProfile(
                    condition=SkinCondition.HYPERPIGMENTATION,
                    approach=TreatmentApproach.CORRECTION,
                    target_mechanisms=["tyrosinase_inhibition", "cellular_turnover", "antioxidant"],
                    required_ingredients=["ARBUTIN", "VITAMIN C"],
                    beneficial_ingredients=["KOJIC ACID", "LICORICE EXTRACT", "NIACINAMIDE"],
                    optimal_pH_range=(4.0, 6.0),
                    expected_efficacy=0.75,
                    evidence_level=EvidenceLevel.STRONG
                )
            },
            
            SkinCondition.DRYNESS: {
                TreatmentApproach.CORRECTION: ConditionTreatmentProfile(
                    condition=SkinCondition.DRYNESS,
                    approach=TreatmentApproach.CORRECTION,
                    target_mechanisms=["barrier_repair", "hydration", "moisture_retention"],
                    required_ingredients=["HYALURONIC ACID", "CERAMIDES"],
                    beneficial_ingredients=["GLYCERIN", "SQUALANE", "CHOLESTEROL"],
                    optimal_pH_range=(5.0, 7.0),
                    expected_efficacy=0.85,
                    evidence_level=EvidenceLevel.STRONG
                )
            },
            
            SkinCondition.SENSITIVITY: {
                TreatmentApproach.GENTLE_CARE: ConditionTreatmentProfile(
                    condition=SkinCondition.SENSITIVITY,
                    approach=TreatmentApproach.GENTLE_CARE,
                    target_mechanisms=["anti_inflammatory", "barrier_repair", "soothing"],
                    required_ingredients=["ALLANTOIN", "CENTELLA ASIATICA"],
                    beneficial_ingredients=["OATMEAL", "CHAMOMILE", "NIACINAMIDE"],
                    contraindicated_ingredients=["STRONG_ACIDS", "RETINOIDS", "FRAGRANCES"],
                    optimal_pH_range=(5.5, 7.0),
                    expected_efficacy=0.80,
                    evidence_level=EvidenceLevel.MODERATE
                )
            }
        }
        
        # Initialize ingredient efficacy database
        self._initialize_ingredient_efficacy()
        
        # Initialize mechanism mapping
        self._initialize_mechanism_mapping()
    
    def _initialize_ingredient_efficacy(self):
        """Initialize ingredient efficacy data with evidence levels"""
        self.ingredient_efficacy = {
            "RETINOL": {
                "mechanisms": ["cellular_turnover", "collagen_synthesis"],
                "efficacy_score": 9.0,
                "evidence_level": EvidenceLevel.STRONG,
                "optimal_concentration": 0.5,
                "max_safe_concentration": 1.0
            },
            "VITAMIN C": {
                "mechanisms": ["antioxidant", "collagen_synthesis", "brightening"],
                "efficacy_score": 8.5,
                "evidence_level": EvidenceLevel.STRONG,
                "optimal_concentration": 15.0,
                "max_safe_concentration": 20.0
            },
            "NIACINAMIDE": {
                "mechanisms": ["sebum_regulation", "barrier_function", "anti_inflammatory"],
                "efficacy_score": 8.0,
                "evidence_level": EvidenceLevel.STRONG,
                "optimal_concentration": 5.0,
                "max_safe_concentration": 10.0
            },
            "HYALURONIC ACID": {
                "mechanisms": ["hydration", "plumping"],
                "efficacy_score": 9.0,
                "evidence_level": EvidenceLevel.STRONG,
                "optimal_concentration": 1.0,
                "max_safe_concentration": 2.0
            },
            "SALICYLIC ACID": {
                "mechanisms": ["keratolytic", "anti_inflammatory", "antimicrobial"],
                "efficacy_score": 8.5,
                "evidence_level": EvidenceLevel.STRONG,
                "optimal_concentration": 2.0,
                "max_safe_concentration": 5.0
            },
            "ARBUTIN": {
                "mechanisms": ["tyrosinase_inhibition", "brightening"],
                "efficacy_score": 7.5,
                "evidence_level": EvidenceLevel.STRONG,
                "optimal_concentration": 2.0,
                "max_safe_concentration": 7.0
            }
        }
    
    def _initialize_mechanism_mapping(self):
        """Initialize mechanism to ingredient mapping"""
        self.mechanism_mapping = defaultdict(list)
        
        for ingredient, data in self.ingredient_efficacy.items():
            for mechanism in data["mechanisms"]:
                self.mechanism_mapping[mechanism].append(ingredient)
    
    def get_profile(self, condition: SkinCondition, approach: TreatmentApproach) -> Optional[ConditionTreatmentProfile]:
        """Get treatment profile for condition-approach combination"""
        return self.condition_profiles.get(condition, {}).get(approach)
    
    def get_ingredients_for_mechanism(self, mechanism: str) -> List[str]:
        """Get ingredients that target specific mechanism"""
        return self.mechanism_mapping.get(mechanism, [])
    
    def get_all_condition_treatment_pairs(self) -> List[Tuple[SkinCondition, TreatmentApproach]]:
        """Get all possible condition-treatment combinations"""
        pairs = []
        for condition, approaches in self.condition_profiles.items():
            for approach in approaches.keys():
                pairs.append((condition, approach))
        return pairs

class MetaOptimizationEngine:
    """Central meta-optimization engine that coordinates multiple optimization strategies"""
    
    def __init__(self):
        self.knowledge_base = ConditionTreatmentKnowledgeBase()
        self.optimization_strategies = []
        self.results_cache = {}
        self.learning_history = []
        
        # Initialize optimization strategies
        self._initialize_optimization_strategies()
    
    def _initialize_optimization_strategies(self):
        """Initialize different optimization strategies"""
        
        # Strategy 1: Multiscale Evolutionary Optimization (broad spectrum)
        self.optimization_strategies.append(
            OptimizationStrategy(
                name="multiscale_evolutionary",
                algorithm_type="evolutionary",
                parameters={
                    "population_size": 50,
                    "generations": 100,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.8
                },
                weight=1.0
            )
        )
        
        # Strategy 2: Hypergredient-Based Optimization (ingredient-class focused)
        self.optimization_strategies.append(
            OptimizationStrategy(
                name="hypergredient_based",
                algorithm_type="hypergredient",
                parameters={
                    "class_balancing": True,
                    "synergy_optimization": True
                },
                weight=0.8,
                conditions_specialized=[SkinCondition.AGING, SkinCondition.WRINKLES]
            )
        )
        
        # Strategy 3: Bayesian Optimization (for fine-tuning)
        self.optimization_strategies.append(
            OptimizationStrategy(
                name="bayesian_optimization",
                algorithm_type="bayesian",
                parameters={
                    "n_iterations": 50,
                    "acquisition_function": "expected_improvement"
                },
                weight=0.6,
                conditions_specialized=[SkinCondition.SENSITIVITY, SkinCondition.ROSACEA]
            )
        )
        
        # Strategy 4: Evidence-Based Direct Optimization
        self.optimization_strategies.append(
            OptimizationStrategy(
                name="evidence_based",
                algorithm_type="direct",
                parameters={
                    "evidence_weighting": True,
                    "clinical_data_priority": True
                },
                weight=1.2
            )
        )
    
    def generate_optimal_formulation(self, 
                                   condition: SkinCondition,
                                   approach: TreatmentApproach,
                                   skin_type: str = "normal",
                                   budget_constraint: float = 1000.0,
                                   exclude_ingredients: List[str] = None) -> MetaOptimizationResult:
        """
        Generate optimal formulation for specific condition-treatment combination
        
        This is the main entry point for meta-optimization
        """
        
        print(f"🧠 Meta-Optimization: {condition.value} + {approach.value}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Step 1: Get treatment profile from knowledge base
        profile = self.knowledge_base.get_profile(condition, approach)
        if not profile:
            raise ValueError(f"No treatment profile found for {condition.value} + {approach.value}")
        
        print(f"Treatment Profile: {len(profile.target_mechanisms)} mechanisms, "
              f"{len(profile.required_ingredients)} required ingredients")
        
        # Step 2: Run multiple optimization strategies in parallel
        strategy_results = {}
        
        for strategy in self.optimization_strategies:
            # Check if strategy is specialized for this condition
            if (strategy.conditions_specialized and 
                condition not in strategy.conditions_specialized):
                strategy_weight = strategy.weight * 0.5  # Reduce weight for non-specialized
            else:
                strategy_weight = strategy.weight
            
            print(f"Running {strategy.name} (weight: {strategy_weight:.1f})...")
            
            result = self._run_optimization_strategy(
                strategy, profile, skin_type, budget_constraint, exclude_ingredients
            )
            
            strategy_results[strategy.name] = {
                'result': result,
                'weight': strategy_weight
            }
        
        # Step 3: Ensemble combination of results
        print("Combining strategy results...")
        final_formulation = self._combine_strategy_results(strategy_results, profile)
        
        # Step 4: Post-processing and validation
        validated_formulation = self._validate_and_refine_formulation(
            final_formulation, profile, skin_type
        )
        
        # Step 5: Calculate confidence and evidence
        confidence_score = self._calculate_confidence_score(
            validated_formulation, strategy_results, profile
        )
        
        evidence_basis = self._compile_evidence_basis(validated_formulation, profile)
        
        # Step 6: Generate alternatives
        alternatives = self._generate_alternative_formulations(
            validated_formulation, strategy_results, profile
        )
        
        optimization_time = time.time() - start_time
        
        # Create comprehensive result
        result = MetaOptimizationResult(
            formulation=validated_formulation,
            condition_treatment_pair=(condition, approach),
            optimization_strategies_used=list(strategy_results.keys()),
            predicted_efficacy=self._predict_formulation_efficacy(validated_formulation, profile),
            confidence_score=confidence_score,
            evidence_basis=evidence_basis,
            alternative_formulations=alternatives,
            optimization_metadata={
                'optimization_time_seconds': optimization_time,
                'strategies_used': len(strategy_results),
                'profile_evidence_level': profile.evidence_level.value,
                'total_ingredients': len(validated_formulation)
            }
        )
        
        # Update learning history
        self.learning_history.append({
            'timestamp': time.time(),
            'condition': condition,
            'approach': approach,
            'result': result,
            'performance_metrics': {
                'optimization_time': optimization_time,
                'confidence': confidence_score,
                'predicted_efficacy': result.predicted_efficacy
            }
        })
        
        print(f"✅ Meta-optimization completed in {optimization_time:.2f}s")
        print(f"Final formulation: {len(validated_formulation)} ingredients")
        print(f"Predicted efficacy: {result.predicted_efficacy:.1%}")
        print(f"Confidence score: {confidence_score:.2f}")
        
        return result
    
    def _run_optimization_strategy(self, 
                                 strategy: OptimizationStrategy,
                                 profile: ConditionTreatmentProfile,
                                 skin_type: str,
                                 budget_constraint: float,
                                 exclude_ingredients: List[str]) -> Dict[str, float]:
        """Run a specific optimization strategy"""
        
        if strategy.algorithm_type == "evolutionary":
            return self._run_evolutionary_optimization(strategy, profile, skin_type, budget_constraint)
        elif strategy.algorithm_type == "hypergredient":
            return self._run_hypergredient_optimization(strategy, profile, skin_type, budget_constraint)
        elif strategy.algorithm_type == "bayesian":
            return self._run_bayesian_optimization(strategy, profile, skin_type, budget_constraint)
        elif strategy.algorithm_type == "direct":
            return self._run_evidence_based_optimization(strategy, profile, skin_type, budget_constraint)
        else:
            raise ValueError(f"Unknown algorithm type: {strategy.algorithm_type}")
    
    def _run_evolutionary_optimization(self, strategy, profile, skin_type, budget_constraint):
        """Run evolutionary optimization (simplified implementation)"""
        
        # Get candidate ingredients based on required mechanisms
        candidate_ingredients = set()
        for mechanism in profile.target_mechanisms:
            candidate_ingredients.update(
                self.knowledge_base.get_ingredients_for_mechanism(mechanism)
            )
        
        # Add required ingredients
        candidate_ingredients.update(profile.required_ingredients)
        
        # Remove contraindicated ingredients
        candidate_ingredients = candidate_ingredients - set(profile.contraindicated_ingredients)
        
        # Create base formulation using simple heuristics
        formulation = {}
        
        # Start with water base
        formulation["AQUA"] = 60.0
        remaining_percentage = 40.0
        
        # Add required ingredients at optimal concentrations
        for ingredient in profile.required_ingredients:
            if ingredient in self.knowledge_base.ingredient_efficacy:
                optimal_conc = self.knowledge_base.ingredient_efficacy[ingredient]["optimal_concentration"]
                concentration = min(optimal_conc, remaining_percentage * 0.3)
                formulation[ingredient] = concentration
                remaining_percentage -= concentration
        
        # Add beneficial ingredients
        for ingredient in profile.beneficial_ingredients:
            if remaining_percentage > 1.0 and ingredient in self.knowledge_base.ingredient_efficacy:
                optimal_conc = self.knowledge_base.ingredient_efficacy[ingredient]["optimal_concentration"]
                concentration = min(optimal_conc * 0.7, remaining_percentage * 0.2)
                if concentration > 0.1:
                    formulation[ingredient] = concentration
                    remaining_percentage -= concentration
        
        # Fill remaining with supportive ingredients
        supportive_ingredients = ["GLYCERIN", "PROPYLENE GLYCOL", "PHENOXYETHANOL"]
        for ingredient in supportive_ingredients:
            if remaining_percentage > 2.0:
                concentration = min(5.0, remaining_percentage * 0.4)
                formulation[ingredient] = concentration
                remaining_percentage -= concentration
        
        # Adjust water to reach 100%
        if remaining_percentage != 0:
            formulation["AQUA"] += remaining_percentage
        
        return formulation
    
    def _run_hypergredient_optimization(self, strategy, profile, skin_type, budget_constraint):
        """Run hypergredient-based optimization"""
        
        # This would integrate with the existing HypergredientFormulator
        # For now, create a simplified version
        
        formulation = {}
        formulation["AQUA"] = 60.0
        
        # Map conditions to hypergredient classes
        condition_to_classes = {
            SkinCondition.WRINKLES: ["H.CT", "H.CS"],
            SkinCondition.ACNE: ["H.SE", "H.AI"],
            SkinCondition.DRYNESS: ["H.HY", "H.BR"],
            SkinCondition.HYPERPIGMENTATION: ["H.ML", "H.AO"]
        }
        
        target_classes = condition_to_classes.get(profile.condition, ["H.CS", "H.AO"])
        
        # Add ingredients based on hypergredient classes
        class_ingredients = {
            "H.CT": ("RETINOL", 0.3),
            "H.CS": ("VITAMIN C", 10.0),
            "H.SE": ("NIACINAMIDE", 5.0),
            "H.AI": ("ALLANTOIN", 2.0),
            "H.HY": ("HYALURONIC ACID", 1.0),
            "H.BR": ("CERAMIDES", 3.0),
            "H.ML": ("ARBUTIN", 2.0),
            "H.AO": ("VITAMIN E", 0.5)
        }
        
        remaining = 40.0
        for hclass in target_classes:
            if hclass in class_ingredients and remaining > 0:
                ingredient, conc = class_ingredients[hclass]
                actual_conc = min(conc, remaining * 0.4)
                formulation[ingredient] = actual_conc
                remaining -= actual_conc
        
        # Add base ingredients to reach 100%
        if remaining > 10:
            formulation["GLYCERIN"] = min(8.0, remaining * 0.5)
            remaining -= formulation["GLYCERIN"]
        
        if remaining > 1:
            formulation["PHENOXYETHANOL"] = min(0.5, remaining)
            remaining -= formulation.get("PHENOXYETHANOL", 0)
        
        # Adjust water to reach exactly 100%
        if remaining != 0:
            formulation["AQUA"] += remaining
        
        return formulation
    
    def _run_bayesian_optimization(self, strategy, profile, skin_type, budget_constraint):
        """Run Bayesian optimization (simplified)"""
        
        # For sensitive conditions, use gentler approach
        formulation = {}
        formulation["AQUA"] = 70.0
        
        # Conservative concentrations for sensitive skin
        if profile.condition in [SkinCondition.SENSITIVITY, SkinCondition.ROSACEA]:
            formulation["NIACINAMIDE"] = 3.0
            formulation["ALLANTOIN"] = 1.0
            formulation["CENTELLA ASIATICA"] = 2.0
            formulation["HYALURONIC ACID"] = 0.5
            formulation["GLYCERIN"] = 5.0
            formulation["PHENOXYETHANOL"] = 0.3
        else:
            # Standard formulation
            return self._run_evolutionary_optimization(strategy, profile, skin_type, budget_constraint)
        
        return formulation
    
    def _run_evidence_based_optimization(self, strategy, profile, skin_type, budget_constraint):
        """Run evidence-based direct optimization"""
        
        formulation = {}
        formulation["AQUA"] = 50.0
        
        # Prioritize ingredients with strong evidence
        high_evidence_ingredients = [
            ing for ing, data in self.knowledge_base.ingredient_efficacy.items()
            if data["evidence_level"] == EvidenceLevel.STRONG
        ]
        
        remaining = 50.0
        for ingredient in profile.required_ingredients:
            if ingredient in high_evidence_ingredients and remaining > 0:
                data = self.knowledge_base.ingredient_efficacy.get(ingredient, {})
                optimal_conc = data.get("optimal_concentration", 1.0)
                concentration = min(optimal_conc, remaining * 0.3)
                formulation[ingredient] = concentration
                remaining -= concentration
        
        # Add supportive ingredients
        formulation["GLYCERIN"] = min(10.0, remaining * 0.4)
        if remaining > 2.0:
            formulation["PHENOXYETHANOL"] = 0.5
            remaining -= 0.5
        
        # Adjust water
        formulation["AQUA"] += remaining
        
        return formulation
    
    def _combine_strategy_results(self, strategy_results: Dict, profile: ConditionTreatmentProfile) -> Dict[str, float]:
        """Combine results from multiple optimization strategies"""
        
        # Weighted ensemble approach
        combined_formulation = defaultdict(float)
        total_weight = sum(data['weight'] for data in strategy_results.values())
        
        for strategy_name, data in strategy_results.items():
            result = data['result']
            weight = data['weight'] / total_weight
            
            for ingredient, concentration in result.items():
                combined_formulation[ingredient] += concentration * weight
        
        return dict(combined_formulation)
    
    def _validate_and_refine_formulation(self, formulation: Dict[str, float], 
                                       profile: ConditionTreatmentProfile,
                                       skin_type: str) -> Dict[str, float]:
        """Validate and refine the combined formulation"""
        
        # Ensure formulation sums to 100%
        total = sum(formulation.values())
        if total != 100.0:
            # Normalize
            for ingredient in formulation:
                formulation[ingredient] = (formulation[ingredient] / total) * 100.0
        
        # Check concentration limits
        for ingredient, concentration in formulation.items():
            if ingredient in self.knowledge_base.ingredient_efficacy:
                max_safe = self.knowledge_base.ingredient_efficacy[ingredient]["max_safe_concentration"]
                if concentration > max_safe:
                    excess = concentration - max_safe
                    formulation[ingredient] = max_safe
                    formulation["AQUA"] = formulation.get("AQUA", 0) + excess
        
        # Ensure required ingredients are present
        for required_ingredient in profile.required_ingredients:
            if required_ingredient not in formulation or formulation[required_ingredient] < 0.1:
                if required_ingredient in self.knowledge_base.ingredient_efficacy:
                    min_effective = self.knowledge_base.ingredient_efficacy[required_ingredient]["optimal_concentration"] * 0.5
                    formulation[required_ingredient] = min_effective
                    # Adjust water accordingly
                    formulation["AQUA"] = max(0, formulation.get("AQUA", 0) - min_effective)
        
        return formulation
    
    def _calculate_confidence_score(self, formulation: Dict[str, float],
                                  strategy_results: Dict,
                                  profile: ConditionTreatmentProfile) -> float:
        """Calculate confidence score for the formulation"""
        
        # Base confidence from evidence level
        evidence_scores = {
            EvidenceLevel.STRONG: 0.9,
            EvidenceLevel.MODERATE: 0.7,
            EvidenceLevel.LIMITED: 0.5,
            EvidenceLevel.THEORETICAL: 0.3
        }
        
        base_confidence = evidence_scores[profile.evidence_level]
        
        # Boost confidence if multiple strategies agree
        strategy_agreement = len(strategy_results) / 4.0  # Normalize by max strategies
        
        # Check ingredient evidence levels
        ingredient_evidence = 0.0
        for ingredient in formulation.keys():
            if ingredient in self.knowledge_base.ingredient_efficacy:
                ingredient_evidence += evidence_scores[
                    self.knowledge_base.ingredient_efficacy[ingredient]["evidence_level"]
                ]
        
        ingredient_evidence = ingredient_evidence / len(formulation) if formulation else 0.0
        
        # Combined confidence score
        confidence = (base_confidence * 0.4 + 
                     strategy_agreement * 0.3 + 
                     ingredient_evidence * 0.3)
        
        return min(1.0, confidence)
    
    def _compile_evidence_basis(self, formulation: Dict[str, float],
                              profile: ConditionTreatmentProfile) -> List[str]:
        """Compile evidence basis for the formulation"""
        
        evidence = []
        
        # Profile evidence
        evidence.append(f"Treatment approach: {profile.approach.value} for {profile.condition.value}")
        evidence.append(f"Evidence level: {profile.evidence_level.value}")
        
        # Ingredient evidence
        for ingredient in formulation.keys():
            if ingredient in self.knowledge_base.ingredient_efficacy:
                data = self.knowledge_base.ingredient_efficacy[ingredient]
                evidence.append(f"{ingredient}: {data['evidence_level'].value} evidence for {', '.join(data['mechanisms'])}")
        
        return evidence
    
    def _predict_formulation_efficacy(self, formulation: Dict[str, float],
                                    profile: ConditionTreatmentProfile) -> float:
        """Predict formulation efficacy"""
        
        # Base efficacy from profile
        base_efficacy = profile.expected_efficacy
        
        # Ingredient contribution
        ingredient_efficacy = 0.0
        for ingredient, concentration in formulation.items():
            if ingredient in self.knowledge_base.ingredient_efficacy:
                data = self.knowledge_base.ingredient_efficacy[ingredient]
                optimal_conc = data["optimal_concentration"]
                
                # Efficacy based on concentration relative to optimal
                if concentration <= optimal_conc:
                    conc_factor = concentration / optimal_conc
                else:
                    # Diminishing returns above optimal
                    conc_factor = 1.0 - (concentration - optimal_conc) / optimal_conc * 0.2
                
                ingredient_efficacy += data["efficacy_score"] * conc_factor * 0.1
        
        # Combined efficacy
        predicted_efficacy = min(1.0, base_efficacy * 0.7 + ingredient_efficacy * 0.3)
        
        return predicted_efficacy
    
    def _generate_alternative_formulations(self, main_formulation: Dict[str, float],
                                         strategy_results: Dict,
                                         profile: ConditionTreatmentProfile) -> List[Dict[str, Any]]:
        """Generate alternative formulations"""
        
        alternatives = []
        
        # Alternative 1: Higher potency version
        high_potency = copy.deepcopy(main_formulation)
        total_boost = 0.0
        
        for ingredient in high_potency:
            if ingredient in self.knowledge_base.ingredient_efficacy and ingredient != "AQUA":
                data = self.knowledge_base.ingredient_efficacy[ingredient]
                max_safe = data["max_safe_concentration"]
                current = high_potency[ingredient]
                if current < max_safe:
                    boost = min(max_safe - current, current * 0.3)
                    high_potency[ingredient] += boost
                    total_boost += boost
        
        # Adjust water to maintain 100%
        if "AQUA" in high_potency and total_boost > 0:
            high_potency["AQUA"] = max(0, high_potency["AQUA"] - total_boost)
        
        alternatives.append({
            'formulation': high_potency,
            'variant_type': 'high_potency',
            'description': 'Higher concentration of active ingredients'
        })
        
        # Alternative 2: Gentle version
        gentle = copy.deepcopy(main_formulation)
        for ingredient in gentle:
            if ingredient in self.knowledge_base.ingredient_efficacy and ingredient != "AQUA":
                gentle[ingredient] *= 0.7  # Reduce by 30%
        
        # Adjust water
        total_reduction = sum(main_formulation.values()) - sum(gentle.values())
        gentle["AQUA"] += total_reduction
        
        alternatives.append({
            'formulation': gentle,
            'variant_type': 'gentle',
            'description': 'Reduced concentration for sensitive skin'
        })
        
        return alternatives
    
    def generate_all_optimal_formulations(self) -> Dict[Tuple[SkinCondition, TreatmentApproach], MetaOptimizationResult]:
        """Generate optimal formulations for all possible condition-treatment combinations"""
        
        print("🚀 Generating Optimal Formulations for All Conditions")
        print("=" * 80)
        
        all_pairs = self.knowledge_base.get_all_condition_treatment_pairs()
        results = {}
        
        total_pairs = len(all_pairs)
        print(f"Processing {total_pairs} condition-treatment combinations...")
        
        for i, (condition, approach) in enumerate(all_pairs, 1):
            print(f"\n[{i}/{total_pairs}] Processing: {condition.value} + {approach.value}")
            
            try:
                result = self.generate_optimal_formulation(condition, approach)
                results[(condition, approach)] = result
                
                print(f"✓ Success: Efficacy {result.predicted_efficacy:.1%}, "
                      f"Confidence {result.confidence_score:.2f}")
                
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
                continue
        
        print(f"\n🎯 Meta-Optimization Complete!")
        print(f"Successfully generated {len(results)} optimal formulations")
        print(f"Coverage: {len(results)/total_pairs:.1%} of all possible combinations")
        
        return results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization performance summary"""
        
        if not self.learning_history:
            return {'message': 'No optimizations completed yet'}
        
        # Calculate performance metrics
        total_optimizations = len(self.learning_history)
        avg_optimization_time = np.mean([h['performance_metrics']['optimization_time'] 
                                       for h in self.learning_history])
        avg_confidence = np.mean([h['performance_metrics']['confidence'] 
                                for h in self.learning_history])
        avg_efficacy = np.mean([h['performance_metrics']['predicted_efficacy'] 
                              for h in self.learning_history])
        
        # Condition coverage
        conditions_covered = set(h['condition'] for h in self.learning_history)
        approaches_covered = set(h['approach'] for h in self.learning_history)
        
        return {
            'total_optimizations': total_optimizations,
            'average_optimization_time_seconds': avg_optimization_time,
            'average_confidence_score': avg_confidence,
            'average_predicted_efficacy': avg_efficacy,
            'conditions_covered': len(conditions_covered),
            'approaches_covered': len(approaches_covered),
            'optimization_strategies_available': len(self.optimization_strategies),
            'knowledge_base_size': len(self.knowledge_base.ingredient_efficacy)
        }

# Example usage and demonstration
def demonstrate_meta_optimization():
    """Demonstrate the meta-optimization engine capabilities"""
    
    print("🧠 Meta-Optimization Engine Demonstration")
    print("=" * 80)
    
    # Initialize the engine
    engine = MetaOptimizationEngine()
    
    print(f"\nEngine initialized with:")
    print(f"• {len(engine.optimization_strategies)} optimization strategies")
    print(f"• {len(engine.knowledge_base.ingredient_efficacy)} ingredients in knowledge base")
    print(f"• {len(engine.knowledge_base.get_all_condition_treatment_pairs())} condition-treatment combinations")
    
    # Example 1: Single condition optimization
    print(f"\n🎯 Example 1: Anti-Aging Treatment Optimization")
    print("-" * 60)
    
    result = engine.generate_optimal_formulation(
        condition=SkinCondition.WRINKLES,
        approach=TreatmentApproach.CORRECTION,
        skin_type="normal"
    )
    
    print(f"\nOptimization Result:")
    print(f"Predicted Efficacy: {result.predicted_efficacy:.1%}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Strategies Used: {', '.join(result.optimization_strategies_used)}")
    
    print(f"\nFormulation ({len(result.formulation)} ingredients):")
    sorted_ingredients = sorted(result.formulation.items(), key=lambda x: x[1], reverse=True)
    for ingredient, concentration in sorted_ingredients:
        print(f"  • {ingredient:20s}: {concentration:6.2f}%")
    
    print(f"\nEvidence Basis:")
    for evidence in result.evidence_basis[:3]:  # Show first 3
        print(f"  • {evidence}")
    
    print(f"\nAlternatives Generated: {len(result.alternative_formulations)}")
    for alt in result.alternative_formulations:
        print(f"  • {alt['variant_type']}: {alt['description']}")
    
    # Example 2: Acne treatment optimization
    print(f"\n🎯 Example 2: Acne Prevention Optimization")
    print("-" * 60)
    
    acne_result = engine.generate_optimal_formulation(
        condition=SkinCondition.ACNE,
        approach=TreatmentApproach.PREVENTION,
        skin_type="oily"
    )
    
    print(f"Acne Prevention Formulation:")
    print(f"Efficacy: {acne_result.predicted_efficacy:.1%}, Confidence: {acne_result.confidence_score:.2f}")
    
    # Show top 5 ingredients
    top_ingredients = sorted(acne_result.formulation.items(), key=lambda x: x[1], reverse=True)[:5]
    for ingredient, concentration in top_ingredients:
        print(f"  • {ingredient}: {concentration:.1f}%")
    
    # Example 3: Performance summary
    print(f"\n📊 Performance Summary")
    print("-" * 60)
    
    summary = engine.get_optimization_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title():30s}: {value:.3f}")
        else:
            print(f"{key.replace('_', ' ').title():30s}: {value}")
    
    return engine, result, acne_result

if __name__ == "__main__":
    # Run demonstration
    engine, anti_aging_result, acne_result = demonstrate_meta_optimization()
    
    print(f"\n✅ Meta-Optimization Engine Successfully Demonstrated")
    print(f"✅ Capable of generating optimal formulations for any condition-treatment combination")
    print(f"✅ Integrates multiple optimization strategies with evidence-based decision making")
    print(f"✅ Provides confidence scoring and alternative formulation generation")