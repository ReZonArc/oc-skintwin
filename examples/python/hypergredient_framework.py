#!/usr/bin/env python3
#
# hypergredient_framework.py
#
# 🧬 Revolutionary Hypergredient Framework Architecture
# A high-level abstraction layer for cosmetic formulation design that groups
# ingredients by functional classes and provides intelligent optimization
# algorithms for formulation generation.
#
# Key Features:
# - Hypergredient taxonomy (H.CT, H.CS, H.AO, H.BR, H.ML, H.HY, H.AI, H.MB, H.SE, H.PD)
# - Dynamic scoring system based on efficacy, bioavailability, safety, cost
# - Interaction matrix for ingredient compatibility and synergies
# - Multi-objective optimization algorithms
# - Real-time formulation generation and evaluation
#
# Part of the OpenCog Multiscale Constraint Optimization system
# --------------------------------------------------------------

import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
from collections import defaultdict

# Import existing modules
try:
    from inci_optimizer import INCISearchSpaceReducer, FormulationConstraint, RegionType
    from multiscale_optimizer import MultiscaleConstraintOptimizer, FormulationCandidate, ObjectiveType
except ImportError:
    # For standalone testing
    class RegionType(Enum):
        EU = "EU"
        FDA = "FDA"
        INTERNATIONAL = "INTERNATIONAL"
    
    class ObjectiveType(Enum):
        EFFICACY = "efficacy"
        SAFETY = "safety"
        COST = "cost"
        STABILITY = "stability"

class HypergredientClass(Enum):
    """Core Hypergredient Classes for functional ingredient categorization"""
    CT = "H.CT"  # Cellular Turnover Agents
    CS = "H.CS"  # Collagen Synthesis Promoters
    AO = "H.AO"  # Antioxidant Systems
    BR = "H.BR"  # Barrier Repair Complex
    ML = "H.ML"  # Melanin Modulators
    HY = "H.HY"  # Hydration Systems
    AI = "H.AI"  # Anti-Inflammatory Agents
    MB = "H.MB"  # Microbiome Balancers
    SE = "H.SE"  # Sebum Regulators
    PD = "H.PD"  # Penetration/Delivery Enhancers

@dataclass
class HypergredientInfo:
    """Comprehensive information about a hypergredient ingredient"""
    inci_name: str
    common_name: str
    hypergredient_class: HypergredientClass
    primary_function: str
    secondary_functions: List[str] = field(default_factory=list)
    
    # Performance metrics (0-10 scale)
    potency: float = 5.0
    efficacy_score: float = 5.0
    bioavailability: float = 50.0  # percentage
    safety_score: float = 5.0
    
    # Physical/chemical properties
    ph_min: float = 3.0
    ph_max: float = 9.0
    stability_score: float = 5.0  # 1-10 scale
    
    # Economic factors
    cost_per_gram: float = 100.0  # ZAR
    
    # Regulatory info
    max_concentration: Dict[RegionType, float] = field(default_factory=dict)
    restrictions: List[str] = field(default_factory=list)
    
    # Interaction data
    incompatible_with: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    
    # Evidence level
    clinical_evidence: str = "Moderate"  # "Strong", "Moderate", "Limited"
    
    def calculate_composite_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate composite score based on weighted metrics"""
        if weights is None:
            weights = {
                'efficacy': 0.35,
                'safety': 0.25,
                'stability': 0.20,
                'cost': 0.15,
                'bioavailability': 0.05
            }
        
        # Normalize cost (lower cost = higher score)
        cost_score = max(0, 10 - (self.cost_per_gram / 100))
        bioavail_score = self.bioavailability / 10  # Convert percentage to 0-10 scale
        
        score = (
            self.efficacy_score * weights.get('efficacy', 0.35) +
            self.safety_score * weights.get('safety', 0.25) +
            self.stability_score * weights.get('stability', 0.20) +
            cost_score * weights.get('cost', 0.15) +
            bioavail_score * weights.get('bioavailability', 0.05)
        )
        
        return min(10.0, max(0.0, score))

class HypergredientDatabase:
    """Dynamic database of hypergredients with optimization capabilities"""
    
    def __init__(self):
        self.hypergredients: Dict[str, HypergredientInfo] = {}
        self.interaction_matrix: Dict[Tuple[str, str], float] = {}
        self.class_mapping: Dict[HypergredientClass, List[str]] = defaultdict(list)
        self._initialize_database()
        self._build_interaction_matrix()
    
    def _initialize_database(self):
        """Initialize the hypergredient database with comprehensive ingredient data"""
        
        # H.CT - Cellular Turnover Agents
        hypergredients_data = [
            # Cellular Turnover Agents
            HypergredientInfo(
                inci_name="TRETINOIN",
                common_name="Tretinoin",
                hypergredient_class=HypergredientClass.CT,
                primary_function="cellular_turnover",
                secondary_functions=["anti_aging", "acne_treatment"],
                potency=10.0, efficacy_score=9.5, bioavailability=85.0, safety_score=6.0,
                ph_min=5.5, ph_max=6.5, stability_score=3.0,
                cost_per_gram=15.00,
                max_concentration={RegionType.EU: 0.1, RegionType.FDA: 0.1},
                restrictions=["prescription_only", "pregnancy_warning"],
                incompatible_with=["BENZOYL PEROXIDE", "ALPHA HYDROXY ACIDS"],
                clinical_evidence="Strong"
            ),
            HypergredientInfo(
                inci_name="BAKUCHIOL",
                common_name="Bakuchiol",
                hypergredient_class=HypergredientClass.CT,
                primary_function="cellular_turnover",
                secondary_functions=["anti_aging", "antioxidant"],
                potency=7.0, efficacy_score=7.5, bioavailability=70.0, safety_score=9.0,
                ph_min=4.0, ph_max=9.0, stability_score=8.0,
                cost_per_gram=240.00,
                max_concentration={RegionType.EU: 2.0, RegionType.FDA: 2.0},
                synergies=["VITAMIN C", "NIACINAMIDE"],
                clinical_evidence="Moderate"
            ),
            HypergredientInfo(
                inci_name="RETINOL",
                common_name="Retinol",
                hypergredient_class=HypergredientClass.CT,
                primary_function="cellular_turnover",
                secondary_functions=["anti_aging", "skin_smoothing"],
                potency=8.0, efficacy_score=8.0, bioavailability=60.0, safety_score=7.0,
                ph_min=5.5, ph_max=6.5, stability_score=4.0,
                cost_per_gram=180.00,
                max_concentration={RegionType.EU: 1.0, RegionType.FDA: 1.0},
                incompatible_with=["ALPHA HYDROXY ACIDS", "BETA HYDROXY ACIDS"],
                clinical_evidence="Strong"
            ),
            HypergredientInfo(
                inci_name="GLYCOLIC ACID",
                common_name="Glycolic Acid",
                hypergredient_class=HypergredientClass.CT,
                primary_function="cellular_turnover",
                secondary_functions=["exfoliation", "brightening"],
                potency=6.0, efficacy_score=7.0, bioavailability=90.0, safety_score=7.0,
                ph_min=3.5, ph_max=4.5, stability_score=9.0,
                cost_per_gram=45.00,
                max_concentration={RegionType.EU: 10.0, RegionType.FDA: 10.0},
                incompatible_with=["RETINOIDS"],
                clinical_evidence="Strong"
            ),
            
            # Collagen Synthesis Promoters
            HypergredientInfo(
                inci_name="PALMITOYL TRIPEPTIDE-1",
                common_name="Matrixyl 3000",
                hypergredient_class=HypergredientClass.CS,
                primary_function="collagen_synthesis",
                secondary_functions=["anti_aging", "wrinkle_reduction"],
                potency=9.0, efficacy_score=8.5, bioavailability=75.0, safety_score=9.0,
                ph_min=5.0, ph_max=7.0, stability_score=7.0,
                cost_per_gram=120.00,
                max_concentration={RegionType.EU: 5.0, RegionType.FDA: 5.0},
                synergies=["VITAMIN C", "PEPTIDES"],
                clinical_evidence="Strong"
            ),
            HypergredientInfo(
                inci_name="COPPER TRIPEPTIDE-1",
                common_name="Copper Peptides",
                hypergredient_class=HypergredientClass.CS,
                primary_function="collagen_synthesis",
                secondary_functions=["wound_healing", "anti_aging"],
                potency=8.0, efficacy_score=8.0, bioavailability=70.0, safety_score=8.0,
                ph_min=6.0, ph_max=7.5, stability_score=6.0,
                cost_per_gram=390.00,
                max_concentration={RegionType.EU: 1.0, RegionType.FDA: 1.0},
                incompatible_with=["VITAMIN C"],
                clinical_evidence="Strong"
            ),
            HypergredientInfo(
                inci_name="ASCORBIC ACID",
                common_name="Vitamin C (L-AA)",
                hypergredient_class=HypergredientClass.CS,
                primary_function="collagen_synthesis",
                secondary_functions=["antioxidant", "brightening"],
                potency=8.0, efficacy_score=8.5, bioavailability=85.0, safety_score=8.0,
                ph_min=3.0, ph_max=4.0, stability_score=2.0,
                cost_per_gram=85.00,
                max_concentration={RegionType.EU: 20.0, RegionType.FDA: 20.0},
                incompatible_with=["COPPER PEPTIDES", "RETINOIDS"],
                synergies=["VITAMIN E", "FERULIC ACID"],
                clinical_evidence="Strong"
            ),
            
            # Antioxidant Systems
            HypergredientInfo(
                inci_name="ASTAXANTHIN",
                common_name="Astaxanthin",
                hypergredient_class=HypergredientClass.AO,
                primary_function="antioxidant",
                secondary_functions=["anti_inflammatory", "photoprotection"],
                potency=9.0, efficacy_score=9.0, bioavailability=60.0, safety_score=9.0,
                ph_min=3.0, ph_max=8.0, stability_score=4.0,
                cost_per_gram=360.00,
                max_concentration={RegionType.EU: 0.1, RegionType.FDA: 0.1},
                synergies=["VITAMIN E", "VITAMIN C"],
                clinical_evidence="Moderate"
            ),
            HypergredientInfo(
                inci_name="RESVERATROL",
                common_name="Resveratrol",
                hypergredient_class=HypergredientClass.AO,
                primary_function="antioxidant",
                secondary_functions=["anti_aging", "anti_inflammatory"],
                potency=7.0, efficacy_score=7.5, bioavailability=65.0, safety_score=8.5,
                ph_min=4.0, ph_max=7.0, stability_score=6.0,
                cost_per_gram=190.00,
                max_concentration={RegionType.EU: 1.0, RegionType.FDA: 1.0},
                synergies=["FERULIC ACID", "VITAMIN E"],
                clinical_evidence="Moderate"
            ),
            HypergredientInfo(
                inci_name="TOCOPHEROL",
                common_name="Vitamin E",
                hypergredient_class=HypergredientClass.AO,
                primary_function="antioxidant",
                secondary_functions=["moisturizing", "stabilizing"],
                potency=6.0, efficacy_score=7.0, bioavailability=80.0, safety_score=9.5,
                ph_min=3.0, ph_max=9.0, stability_score=9.0,
                cost_per_gram=50.00,
                max_concentration={RegionType.EU: 1.0, RegionType.FDA: 1.0},
                synergies=["VITAMIN C", "FERULIC ACID"],
                clinical_evidence="Strong"
            ),
            
            # Barrier Repair Complex
            HypergredientInfo(
                inci_name="CERAMIDE NP",
                common_name="Ceramide NP",
                hypergredient_class=HypergredientClass.BR,
                primary_function="barrier_repair",
                secondary_functions=["moisturizing", "anti_aging"],
                potency=8.0, efficacy_score=8.5, bioavailability=70.0, safety_score=9.5,
                ph_min=4.0, ph_max=8.0, stability_score=8.0,
                cost_per_gram=450.00,
                max_concentration={RegionType.EU: 5.0, RegionType.FDA: 5.0},
                synergies=["CHOLESTEROL", "FATTY ACIDS"],
                clinical_evidence="Strong"
            ),
            
            # Hydration Systems
            HypergredientInfo(
                inci_name="SODIUM HYALURONATE",
                common_name="Hyaluronic Acid",
                hypergredient_class=HypergredientClass.HY,
                primary_function="hydration",
                secondary_functions=["plumping", "anti_aging"],
                potency=9.0, efficacy_score=9.0, bioavailability=85.0, safety_score=9.5,
                ph_min=4.0, ph_max=9.0, stability_score=9.0,
                cost_per_gram=150.00,
                max_concentration={RegionType.EU: 2.0, RegionType.FDA: 2.0},
                synergies=["GLYCERIN", "PEPTIDES"],
                clinical_evidence="Strong"
            ),
            
            # Melanin Modulators
            HypergredientInfo(
                inci_name="ALPHA ARBUTIN",
                common_name="Alpha Arbutin",
                hypergredient_class=HypergredientClass.ML,
                primary_function="melanin_modulation",
                secondary_functions=["brightening", "hyperpigmentation"],
                potency=7.0, efficacy_score=8.0, bioavailability=75.0, safety_score=9.0,
                ph_min=4.0, ph_max=7.0, stability_score=8.0,
                cost_per_gram=180.00,
                max_concentration={RegionType.EU: 2.0, RegionType.FDA: 2.0},
                synergies=["VITAMIN C", "NIACINAMIDE"],
                clinical_evidence="Strong"
            ),
        ]
        
        # Add hypergredients to database
        for hypergredient in hypergredients_data:
            self.add_hypergredient(hypergredient)
    
    def _build_interaction_matrix(self):
        """Build the interaction matrix for hypergredient compatibility"""
        # Interaction matrix: positive values = synergy, negative = antagonism
        interactions = {
            # Cellular Turnover synergies
            ("H.CT", "H.CS"): 1.5,  # Cellular turnover + collagen synthesis
            ("H.CT", "H.AO"): 0.8,  # Mild antagonism due to oxidation sensitivity
            
            # Collagen synthesis synergies
            ("H.CS", "H.AO"): 2.0,  # Strong synergy (antioxidants protect collagen)
            ("H.CS", "H.HY"): 1.3,  # Good synergy for anti-aging
            
            # Barrier repair synergies
            ("H.BR", "H.HY"): 2.5,  # Excellent synergy for skin health
            ("H.BR", "H.AI"): 1.8,  # Good for sensitive skin
            
            # Antioxidant network effects
            ("H.AO", "H.ML"): 1.8,  # Antioxidants protect brightening agents
            ("H.AO", "H.AI"): 2.2,  # Strong anti-inflammatory synergy
            
            # Hydration synergies
            ("H.HY", "H.CS"): 1.4,  # Hydration supports collagen synthesis
            ("H.HY", "H.BR"): 2.5,  # Barrier + hydration = excellent combination
            
            # Potential antagonisms
            ("H.SE", "H.CT"): 0.6,  # Sebum regulation + strong actives can irritate
            ("H.ML", "H.CT"): 0.9,  # pH conflicts between brightening and turnover agents
        }
        
        # Make interactions symmetric
        for (class1, class2), score in interactions.items():
            self.interaction_matrix[(class1, class2)] = score
            self.interaction_matrix[(class2, class1)] = score
        
        # Default interaction score for unlisted pairs
        for class1 in HypergredientClass:
            for class2 in HypergredientClass:
                key = (class1.value, class2.value)
                if key not in self.interaction_matrix:
                    self.interaction_matrix[key] = 1.0  # Neutral interaction
    
    def add_hypergredient(self, hypergredient: HypergredientInfo):
        """Add a hypergredient to the database"""
        self.hypergredients[hypergredient.inci_name] = hypergredient
        self.class_mapping[hypergredient.hypergredient_class].append(hypergredient.inci_name)
    
    def get_hypergredients_by_class(self, hypergredient_class: HypergredientClass) -> List[HypergredientInfo]:
        """Get all hypergredients belonging to a specific class"""
        return [self.hypergredients[name] for name in self.class_mapping[hypergredient_class]]
    
    def get_interaction_score(self, class1: HypergredientClass, class2: HypergredientClass) -> float:
        """Get interaction score between two hypergredient classes"""
        return self.interaction_matrix.get((class1.value, class2.value), 1.0)
    
    def search_hypergredients(self, 
                            function: str = None,
                            min_efficacy: float = None,
                            max_cost: float = None,
                            exclude_ingredients: List[str] = None) -> List[HypergredientInfo]:
        """Search hypergredients based on criteria"""
        results = []
        exclude_ingredients = exclude_ingredients or []
        
        for hypergredient in self.hypergredients.values():
            if hypergredient.inci_name in exclude_ingredients:
                continue
                
            if function and function not in [hypergredient.primary_function] + hypergredient.secondary_functions:
                continue
                
            if min_efficacy and hypergredient.efficacy_score < min_efficacy:
                continue
                
            if max_cost and hypergredient.cost_per_gram > max_cost:
                continue
                
            results.append(hypergredient)
        
        return sorted(results, key=lambda x: x.calculate_composite_score(), reverse=True)

class HypergredientFormulator:
    """Advanced formulation generator using hypergredient optimization"""
    
    def __init__(self, database: HypergredientDatabase = None):
        self.database = database or HypergredientDatabase()
        self.ml_model = None  # Placeholder for future ML integration
    
    def generate_formulation(self, 
                           target_concerns: List[str],
                           skin_type: str = "normal",
                           budget: float = 1000.0,
                           exclude_ingredients: List[str] = None,
                           texture_preference: str = "lightweight") -> Dict[str, Any]:
        """Generate optimal formulation using hypergredient framework"""
        
        # Map concerns to hypergredient classes
        concern_mapping = {
            'wrinkles': [HypergredientClass.CT, HypergredientClass.CS],
            'firmness': [HypergredientClass.CS, HypergredientClass.HY],
            'brightness': [HypergredientClass.ML, HypergredientClass.AO],
            'hyperpigmentation': [HypergredientClass.ML, HypergredientClass.AO],
            'hydration': [HypergredientClass.HY, HypergredientClass.BR],
            'anti_aging': [HypergredientClass.CT, HypergredientClass.CS, HypergredientClass.AO],
            'sensitivity': [HypergredientClass.AI, HypergredientClass.BR],
            'acne': [HypergredientClass.SE, HypergredientClass.AI],
        }
        
        # Collect required hypergredient classes
        required_classes = set()
        for concern in target_concerns:
            if concern in concern_mapping:
                required_classes.update(concern_mapping[concern])
        
        # Select optimal hypergredients for each class
        selected_hypergredients = {}
        total_cost = 0.0
        
        for hypergredient_class in required_classes:
            candidates = self.database.get_hypergredients_by_class(hypergredient_class)
            
            # Filter by budget and exclusions
            affordable_candidates = [
                c for c in candidates 
                if c.cost_per_gram <= budget/10  # Allow max 10% of budget per ingredient
                and (not exclude_ingredients or c.inci_name not in exclude_ingredients)
            ]
            
            if not affordable_candidates:
                continue
            
            # Score candidates considering skin type
            skin_type_weights = self._get_skin_type_weights(skin_type)
            best_candidate = max(
                affordable_candidates,
                key=lambda x: self._score_hypergredient(x, skin_type_weights)
            )
            
            # Calculate optimal concentration
            optimal_concentration = self._calculate_optimal_concentration(
                best_candidate, hypergredient_class, target_concerns
            )
            
            selected_hypergredients[hypergredient_class.value] = {
                'hypergredient': best_candidate,
                'concentration': optimal_concentration,
                'reasoning': f"Highest efficacy/cost ratio for {hypergredient_class.value}"
            }
            
            total_cost += best_candidate.cost_per_gram * (optimal_concentration / 100)
        
        # Calculate synergy score
        synergy_score = self._calculate_formulation_synergy(selected_hypergredients)
        
        # Generate formulation report
        formulation = {
            'selected_hypergredients': selected_hypergredients,
            'total_cost': total_cost,
            'synergy_score': synergy_score,
            'efficacy_prediction': self._predict_efficacy(selected_hypergredients, target_concerns),
            'stability_timeline': self._estimate_stability(selected_hypergredients),
            'safety_assessment': self._assess_safety(selected_hypergredients, skin_type),
            'recommendations': self._generate_recommendations(selected_hypergredients, target_concerns)
        }
        
        return formulation
    
    def _get_skin_type_weights(self, skin_type: str) -> Dict[str, float]:
        """Get scoring weights based on skin type"""
        skin_weights = {
            'normal': {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost': 0.15, 'bioavailability': 0.05},
            'sensitive': {'efficacy': 0.25, 'safety': 0.45, 'stability': 0.15, 'cost': 0.10, 'bioavailability': 0.05},
            'oily': {'efficacy': 0.40, 'safety': 0.20, 'stability': 0.25, 'cost': 0.10, 'bioavailability': 0.05},
            'dry': {'efficacy': 0.30, 'safety': 0.25, 'stability': 0.20, 'cost': 0.15, 'bioavailability': 0.10},
        }
        return skin_weights.get(skin_type, skin_weights['normal'])
    
    def _score_hypergredient(self, hypergredient: HypergredientInfo, weights: Dict[str, float]) -> float:
        """Score a hypergredient based on weighted criteria"""
        return hypergredient.calculate_composite_score(weights)
    
    def _calculate_optimal_concentration(self, hypergredient: HypergredientInfo, 
                                       class_type: HypergredientClass, 
                                       concerns: List[str]) -> float:
        """Calculate optimal concentration for a hypergredient"""
        base_concentration = {
            HypergredientClass.CT: 1.0,   # Strong actives, low concentration
            HypergredientClass.CS: 3.0,   # Peptides, moderate concentration
            HypergredientClass.AO: 0.5,   # Antioxidants, variable
            HypergredientClass.BR: 2.0,   # Barrier repair, moderate
            HypergredientClass.HY: 1.0,   # Hydration, depends on molecule size
            HypergredientClass.ML: 2.0,   # Brightening, moderate
            HypergredientClass.AI: 1.0,   # Anti-inflammatory, gentle
            HypergredientClass.MB: 1.0,   # Microbiome, low concentration
            HypergredientClass.SE: 2.0,   # Sebum regulation, moderate
            HypergredientClass.PD: 0.5,   # Penetration enhancers, low
        }.get(class_type, 1.0)
        
        # Adjust based on safety and regulatory limits
        max_allowed = min(
            hypergredient.max_concentration.get(RegionType.EU, 10.0),
            hypergredient.max_concentration.get(RegionType.FDA, 10.0),
            base_concentration * 2
        )
        
        return min(base_concentration, max_allowed)
    
    def _calculate_formulation_synergy(self, selected_hypergredients: Dict[str, Any]) -> float:
        """Calculate overall synergy score for the formulation"""
        if not selected_hypergredients or len(selected_hypergredients) < 2:
            return 1.0
        
        classes = list(selected_hypergredients.keys())
        total_synergy = 0.0
        pair_count = 0
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes[i+1:], i+1):
                synergy = self.database.get_interaction_score(
                    HypergredientClass(class1), 
                    HypergredientClass(class2)
                )
                total_synergy += synergy
                pair_count += 1
        
        return total_synergy / max(pair_count, 1) if pair_count > 0 else 1.0
    
    def _predict_efficacy(self, selected_hypergredients: Dict[str, Any], concerns: List[str]) -> float:
        """Predict formulation efficacy"""
        if not selected_hypergredients:
            return 0.0
        
        total_efficacy = 0.0
        for class_name, data in selected_hypergredients.items():
            hypergredient = data['hypergredient']
            concentration = data['concentration']
            
            # Efficacy contribution = base efficacy * concentration factor * synergy
            contribution = hypergredient.efficacy_score * (concentration / 5.0) * 0.8
            total_efficacy += contribution
        
        # Apply synergy multiplier
        synergy_multiplier = min(1.5, self._calculate_formulation_synergy(selected_hypergredients))
        predicted_efficacy = min(100.0, total_efficacy * synergy_multiplier)
        
        return predicted_efficacy
    
    def _estimate_stability(self, selected_hypergredients: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate formulation stability timeline"""
        if not selected_hypergredients:
            return {
                'estimated_shelf_life_months': 12,
                'limiting_factor': 'No ingredients selected',
                'storage_recommendations': ['room_temperature']
            }
        
        min_stability = min(
            data['hypergredient'].stability_score 
            for data in selected_hypergredients.values()
        )
        
        stability_months = {
            10: 36, 9: 24, 8: 18, 7: 12, 6: 9, 5: 6, 4: 4, 3: 3, 2: 2, 1: 1
        }
        
        estimated_months = stability_months.get(int(min_stability), 6)
        
        return {
            'estimated_shelf_life_months': estimated_months,
            'limiting_factor': min(
                selected_hypergredients.values(),
                key=lambda x: x['hypergredient'].stability_score
            )['hypergredient'].inci_name,
            'storage_recommendations': ['cool', 'dry', 'dark'] if min_stability < 7 else ['room_temperature']
        }
    
    def _assess_safety(self, selected_hypergredients: Dict[str, Any], skin_type: str) -> Dict[str, Any]:
        """Assess formulation safety profile"""
        if not selected_hypergredients:
            return {
                'overall_safety_score': 8.0,
                'skin_type_suitability': True,
                'warnings': [],
                'patch_test_recommended': False
            }
        
        min_safety = min(
            data['hypergredient'].safety_score 
            for data in selected_hypergredients.values()
        )
        
        safety_profile = {
            'overall_safety_score': min_safety,
            'skin_type_suitability': min_safety >= 7 or skin_type == 'normal',
            'warnings': [],
            'patch_test_recommended': min_safety < 8
        }
        
        # Check for specific warnings
        for data in selected_hypergredients.values():
            hypergredient = data['hypergredient']
            if hypergredient.restrictions:
                safety_profile['warnings'].extend(hypergredient.restrictions)
        
        return safety_profile
    
    def _generate_recommendations(self, selected_hypergredients: Dict[str, Any], concerns: List[str]) -> List[str]:
        """Generate formulation recommendations"""
        recommendations = []
        
        # Usage recommendations
        has_retinoids = any(
            'CT' in class_name for class_name in selected_hypergredients.keys()
        )
        if has_retinoids:
            recommendations.append("Use only in evening routine")
            recommendations.append("Always use sunscreen during the day")
        
        # Application order
        recommendations.append("Apply to clean, dry skin")
        recommendations.append("Follow with moisturizer if needed")
        
        # Timeline expectations
        if 'anti_aging' in concerns or 'wrinkles' in concerns:
            recommendations.append("Visible results expected in 6-12 weeks")
        if 'hyperpigmentation' in concerns or 'brightness' in concerns:
            recommendations.append("Brightening effects visible in 4-8 weeks")
        
        return recommendations

# Example usage and demonstration
def demonstrate_hypergredient_framework():
    """Demonstrate the hypergredient framework with example formulation"""
    print("🧬 Hypergredient Framework Architecture Demonstration")
    print("=" * 60)
    
    # Initialize framework
    database = HypergredientDatabase()
    formulator = HypergredientFormulator(database)
    
    print(f"\nDatabase initialized with {len(database.hypergredients)} hypergredients")
    print(f"Hypergredient classes: {', '.join([hc.value for hc in HypergredientClass])}")
    
    # Example: Generate anti-aging formulation
    print("\n🎯 Generating Optimal Anti-Aging Formulation")
    print("-" * 40)
    
    request = {
        'target_concerns': ['wrinkles', 'firmness', 'brightness'],
        'skin_type': 'normal',
        'budget': 1500,  # ZAR
        'exclude_ingredients': ['TRETINOIN'],  # Exclude prescription ingredients
        'texture_preference': 'lightweight'
    }
    
    print("Request parameters:")
    for key, value in request.items():
        print(f"  • {key}: {value}")
    
    # Generate formulation
    result = formulator.generate_formulation(**request)
    
    print(f"\n✅ Optimal Formulation Generated")
    print("-" * 30)
    print(f"Total Cost: R{result['total_cost']:.2f}")
    print(f"Synergy Score: {result['synergy_score']:.2f}/10")
    print(f"Predicted Efficacy: {result['efficacy_prediction']:.1f}%")
    
    print(f"\nSelected Hypergredients:")
    for class_name, data in result['selected_hypergredients'].items():
        hypergredient = data['hypergredient']
        concentration = data['concentration']
        print(f"  • {class_name}: {hypergredient.common_name} ({concentration:.1f}%)")
        print(f"    - {data['reasoning']}")
    
    print(f"\nStability Assessment:")
    stability = result['stability_timeline']
    print(f"  • Estimated shelf life: {stability['estimated_shelf_life_months']} months")
    print(f"  • Limiting factor: {stability['limiting_factor']}")
    
    print(f"\nSafety Assessment:")
    safety = result['safety_assessment']
    print(f"  • Overall safety score: {safety['overall_safety_score']:.1f}/10")
    print(f"  • Patch test recommended: {'Yes' if safety['patch_test_recommended'] else 'No'}")
    
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")
    
    return result

if __name__ == "__main__":
    demonstrate_hypergredient_framework()