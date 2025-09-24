# Hypergraph Encoding of Ingredient and Action Ontologies

## Abstract

This document describes the hypergraph-based knowledge representation system for cosmeceutical ingredients and their biological actions, adapted from OpenCog's AtomSpace architecture. The system enables complex relationship modeling, multi-scale biological effect representation, and efficient constraint propagation for formulation optimization.

## 1. Introduction

### 1.1 Hypergraph Fundamentals

A hypergraph extends traditional graph structures by allowing edges (hyperedges) to connect any number of vertices, not just pairs. In our cosmeceutical formulation context:

- **Vertices (Atoms)**: Represent ingredients, biological targets, actions, constraints, and properties
- **Hyperedges (Links)**: Represent complex relationships involving multiple entities
- **Weights**: Encode confidence levels, interaction strengths, and probabilistic relationships

### 1.2 Advantages for Formulation Science

1. **Multi-ingredient Interactions**: Model synergistic and antagonistic effects involving 3+ ingredients
2. **Scale-bridging Relationships**: Connect molecular actions to organ-level outcomes
3. **Constraint Networks**: Represent complex regulatory and safety constraints
4. **Probabilistic Reasoning**: Handle uncertainty in ingredient interactions and biological outcomes

## 2. Atomic Structure Design

### 2.1 Core Atom Types

#### 2.1.1 Ingredient Atoms

```python
class IngredientAtom:
    """Represents a single cosmetic ingredient with its properties"""
    
    def __init__(self, inci_name: str, properties: Dict):
        self.inci_name = inci_name
        self.cas_number = properties.get('cas_number')
        self.molecular_weight = properties.get('molecular_weight')
        self.function_category = properties.get('function')  # e.g., 'moisturizer', 'antioxidant'
        self.safety_profile = properties.get('safety_profile')
        self.regulatory_status = properties.get('regulatory_status')
        self.truth_value = TruthValue(
            strength=properties.get('data_confidence', 0.8),
            count=properties.get('evidence_count', 10)
        )

# Example instantiation
niacinamide = IngredientAtom('NIACINAMIDE', {
    'cas_number': '98-92-0',
    'molecular_weight': 122.12,
    'function': ['anti_aging', 'brightening', 'sebum_control'],
    'safety_profile': {'irritation_potential': 0.1, 'sensitization': 0.05},
    'regulatory_status': {'EU': 'approved', 'FDA': 'approved'},
    'data_confidence': 0.95,
    'evidence_count': 150
})
```

#### 2.1.2 Biological Target Atoms

```python
class BiologicalTargetAtom:
    """Represents biological targets (receptors, enzymes, pathways)"""
    
    def __init__(self, target_name: str, target_type: str, scale: BiologicalScale):
        self.target_name = target_name
        self.target_type = target_type  # 'receptor', 'enzyme', 'pathway', 'structure'
        self.biological_scale = scale
        self.location = None  # cellular, tissue, organ location
        self.function = None  # primary biological function
        
# Examples
nicotinic_acid_receptor = BiologicalTargetAtom(
    'NIACIN_RECEPTOR_GPR109A', 'receptor', BiologicalScale.MOLECULAR
)

collagen_synthesis_pathway = BiologicalTargetAtom(
    'COLLAGEN_SYNTHESIS', 'pathway', BiologicalScale.CELLULAR  
)

stratum_corneum = BiologicalTargetAtom(
    'STRATUM_CORNEUM', 'structure', BiologicalScale.TISSUE
)
```

#### 2.1.3 Action Atoms

```python
class ActionAtom:
    """Represents biological actions and effects"""
    
    def __init__(self, action_name: str, action_type: str, magnitude: float):
        self.action_name = action_name
        self.action_type = action_type  # 'activation', 'inhibition', 'modulation'
        self.magnitude = magnitude  # quantitative effect size
        self.time_course = None  # temporal dynamics
        self.dose_response = None  # concentration-effect relationship

# Examples  
antioxidant_action = ActionAtom('ANTIOXIDANT_ACTIVITY', 'activation', 0.8)
collagen_stimulation = ActionAtom('COLLAGEN_SYNTHESIS_INCREASE', 'activation', 0.6)
melanin_inhibition = ActionAtom('MELANOGENESIS_INHIBITION', 'inhibition', 0.7)
```

#### 2.1.4 Constraint Atoms

```python
class ConstraintAtom:
    """Represents formulation constraints and requirements"""
    
    def __init__(self, constraint_type: str, parameters: Dict):
        self.constraint_type = constraint_type
        self.parameters = parameters
        self.enforcement_level = parameters.get('enforcement', 'hard')  # 'hard', 'soft', 'preference'
        self.source = parameters.get('source')  # 'regulatory', 'safety', 'stability', 'cost'

# Examples
eu_retinol_limit = ConstraintAtom('CONCENTRATION_LIMIT', {
    'ingredient': 'RETINOL',
    'max_concentration': 0.3,
    'region': 'EU',
    'enforcement': 'hard',
    'source': 'regulatory'
})

ph_stability_constraint = ConstraintAtom('PH_RANGE', {
    'min_ph': 5.5,
    'max_ph': 7.0,
    'enforcement': 'soft',
    'source': 'stability'
})
```

### 2.2 Hyperedge Types

#### 2.2.1 Binding Links

Represent molecular-level binding interactions between ingredients and biological targets.

```python
class BindingLink:
    """N-ary relationship for ingredient-target binding"""
    
    def __init__(self, ingredient_atoms: List[IngredientAtom], 
                 target_atom: BiologicalTargetAtom, 
                 binding_affinity: float, 
                 cooperativity: float = 1.0):
        self.ingredients = ingredient_atoms
        self.target = target_atom
        self.binding_affinity = binding_affinity  # Kd or IC50
        self.cooperativity = cooperativity  # for multi-ingredient binding
        self.truth_value = self._calculate_binding_probability()
    
    def _calculate_binding_probability(self) -> TruthValue:
        """Calculate probability of binding based on concentrations and affinity"""
        # Hill equation for cooperative binding
        concentration_sum = sum(ing.concentration for ing in self.ingredients)
        bound_fraction = (concentration_sum ** self.cooperativity) / (
            self.binding_affinity ** self.cooperativity + concentration_sum ** self.cooperativity
        )
        return TruthValue(strength=bound_fraction, count=len(self.ingredients) * 10)

# Example: Niacinamide binding to PARP-1 (DNA repair enzyme)
niacinamide_parp_binding = BindingLink(
    ingredient_atoms=[niacinamide],
    target_atom=BiologicalTargetAtom('PARP1', 'enzyme', BiologicalScale.MOLECULAR),
    binding_affinity=50.0,  # μM
    cooperativity=1.0
)

# Example: Synergistic binding of Vitamin C + Vitamin E to oxidative stress targets
antioxidant_synergy = BindingLink(
    ingredient_atoms=[ascorbic_acid, tocopherol],
    target_atom=BiologicalTargetAtom('ROS_SCAVENGING', 'pathway', BiologicalScale.MOLECULAR),
    binding_affinity=10.0,  # Combined IC50
    cooperativity=1.5  # Positive cooperativity
)
```

#### 2.2.2 Effect Propagation Links

Model how molecular actions propagate across biological scales.

```python
class EffectPropagationLink:
    """Multi-scale effect propagation from molecular to organ level"""
    
    def __init__(self, source_action: ActionAtom, 
                 target_scale: BiologicalScale, 
                 propagation_efficiency: float,
                 intermediate_targets: List[BiologicalTargetAtom] = None):
        self.source_action = source_action
        self.target_scale = target_scale
        self.propagation_efficiency = propagation_efficiency
        self.intermediate_targets = intermediate_targets or []
        self.time_delay = self._calculate_propagation_delay()
    
    def _calculate_propagation_delay(self) -> float:
        """Calculate time for effect to propagate across scales"""
        scale_delays = {
            BiologicalScale.MOLECULAR: 0.1,    # minutes
            BiologicalScale.CELLULAR: 60.0,    # 1 hour  
            BiologicalScale.TISSUE: 1440.0,    # 24 hours
            BiologicalScale.ORGAN: 10080.0     # 1 week
        }
        return scale_delays.get(self.target_scale, 1440.0)

# Example: Retinol molecular binding → cellular gene expression → tissue remodeling
retinol_pathway = EffectPropagationLink(
    source_action=ActionAtom('RETINOIC_ACID_RECEPTOR_ACTIVATION', 'activation', 0.8),
    target_scale=BiologicalScale.TISSUE,
    propagation_efficiency=0.6,
    intermediate_targets=[
        BiologicalTargetAtom('COLLAGEN_GENE_EXPRESSION', 'pathway', BiologicalScale.CELLULAR),
        BiologicalTargetAtom('FIBROBLAST_ACTIVITY', 'process', BiologicalScale.CELLULAR)
    ]
)
```

#### 2.2.3 Interaction Links

Represent complex ingredient-ingredient interactions (synergistic, antagonistic, additive).

```python
class InteractionLink:
    """N-ary ingredient interaction relationship"""
    
    def __init__(self, ingredient_atoms: List[IngredientAtom], 
                 interaction_type: str, 
                 interaction_strength: float,
                 affected_properties: List[str]):
        self.ingredients = ingredient_atoms
        self.interaction_type = interaction_type  # 'synergistic', 'antagonistic', 'additive'
        self.interaction_strength = interaction_strength  # quantitative effect
        self.affected_properties = affected_properties
        self.experimental_evidence = []
    
    def calculate_combined_effect(self, individual_effects: List[float]) -> float:
        """Calculate combined effect based on interaction type"""
        if self.interaction_type == 'additive':
            return sum(individual_effects)
        elif self.interaction_type == 'synergistic':
            return sum(individual_effects) * (1 + self.interaction_strength)
        elif self.interaction_type == 'antagonistic':
            return sum(individual_effects) * (1 - self.interaction_strength)
        else:
            return sum(individual_effects)  # default to additive

# Example: Vitamin C + Ferulic Acid synergistic antioxidant interaction
vitamin_c_ferulic_synergy = InteractionLink(
    ingredient_atoms=[ascorbic_acid, ferulic_acid],
    interaction_type='synergistic',
    interaction_strength=0.8,  # 80% enhancement
    affected_properties=['antioxidant_activity', 'stability', 'penetration']
)

# Example: Retinol + Vitamin C antagonistic interaction (instability)
retinol_vitamin_c_antagonism = InteractionLink(
    ingredient_atoms=[retinol, ascorbic_acid],
    interaction_type='antagonistic', 
    interaction_strength=0.6,  # 60% reduction in stability
    affected_properties=['stability', 'efficacy']
)
```

#### 2.2.4 Constraint Propagation Links

Model how constraints propagate through the formulation network.

```python
class ConstraintPropagationLink:
    """Constraint propagation across ingredient network"""
    
    def __init__(self, source_constraint: ConstraintAtom, 
                 affected_ingredients: List[IngredientAtom],
                 propagation_rules: Dict):
        self.source_constraint = source_constraint
        self.affected_ingredients = affected_ingredients
        self.propagation_rules = propagation_rules
        self.constraint_strength = self._calculate_constraint_strength()
    
    def _calculate_constraint_strength(self) -> float:
        """Calculate how strongly constraint affects each ingredient"""
        if self.source_constraint.enforcement_level == 'hard':
            return 1.0
        elif self.source_constraint.enforcement_level == 'soft':
            return 0.7
        else:  # preference
            return 0.3

# Example: pH constraint affects multiple pH-sensitive ingredients
ph_constraint_propagation = ConstraintPropagationLink(
    source_constraint=ph_stability_constraint,
    affected_ingredients=[ascorbic_acid, retinol, kojic_acid],
    propagation_rules={
        'ascorbic_acid': {'ph_sensitivity': 0.9, 'degradation_rate': 'exponential'},
        'retinol': {'ph_sensitivity': 0.7, 'degradation_rate': 'linear'},
        'kojic_acid': {'ph_sensitivity': 0.5, 'degradation_rate': 'linear'}
    }
)
```

## 3. Hypergraph Construction Algorithm

### 3.1 Automated Ontology Generation

```python
class HypergraphOntologyBuilder:
    """Builds hypergraph ontology from ingredient and biological data"""
    
    def __init__(self):
        self.atoms = {}  # atom_id -> atom
        self.links = {}  # link_id -> link
        self.next_id = 0
        
    def build_ingredient_ontology(self, ingredient_database: Dict) -> Dict:
        """Build ingredient portion of hypergraph"""
        
        for inci_name, ingredient_data in ingredient_database.items():
            # Create ingredient atom
            ingredient_atom = IngredientAtom(inci_name, ingredient_data)
            self.atoms[f"ingredient_{self.next_id}"] = ingredient_atom
            self.next_id += 1
            
            # Create biological target atoms for this ingredient
            for target_name, target_data in ingredient_data.get('targets', {}).items():
                target_atom = BiologicalTargetAtom(
                    target_name, 
                    target_data['type'], 
                    target_data['scale']
                )
                target_id = f"target_{self.next_id}"
                self.atoms[target_id] = target_atom
                self.next_id += 1
                
                # Create binding link
                binding_link = BindingLink(
                    [ingredient_atom], 
                    target_atom, 
                    target_data['affinity']
                )
                self.links[f"binding_{self.next_id}"] = binding_link
                self.next_id += 1
        
        return {'atoms': self.atoms, 'links': self.links}
    
    def discover_interactions(self, experimental_data: Dict) -> List[InteractionLink]:
        """Discover ingredient interactions from experimental data"""
        
        interactions = []
        
        for experiment in experimental_data:
            ingredients = [self.find_ingredient_atom(name) for name in experiment['ingredients']]
            
            if len(ingredients) > 1:
                interaction_type = self._classify_interaction(experiment['results'])
                interaction_strength = self._quantify_interaction(experiment['results'])
                
                interaction_link = InteractionLink(
                    ingredients,
                    interaction_type,
                    interaction_strength,
                    experiment['measured_properties']
                )
                interactions.append(interaction_link)
        
        return interactions
    
    def _classify_interaction(self, results: Dict) -> str:
        """Classify interaction type from experimental results"""
        expected_additive = sum(results['individual_effects'])
        observed_combined = results['combined_effect']
        
        ratio = observed_combined / expected_additive
        
        if ratio > 1.2:
            return 'synergistic'
        elif ratio < 0.8:
            return 'antagonistic'
        else:
            return 'additive'
    
    def _quantify_interaction(self, results: Dict) -> float:
        """Quantify interaction strength"""
        expected_additive = sum(results['individual_effects'])
        observed_combined = results['combined_effect']
        
        return abs((observed_combined - expected_additive) / expected_additive)
```

### 3.2 Scale-Bridging Relationship Discovery

```python
class ScaleBridgeDiscovery:
    """Discovers relationships that bridge biological scales"""
    
    def __init__(self, ontology: HypergraphOntologyBuilder):
        self.ontology = ontology
        self.scale_hierarchy = [
            BiologicalScale.MOLECULAR,
            BiologicalScale.CELLULAR, 
            BiologicalScale.TISSUE,
            BiologicalScale.ORGAN
        ]
    
    def discover_propagation_pathways(self, ingredient_atom: IngredientAtom) -> List[EffectPropagationLink]:
        """Discover how ingredient effects propagate across scales"""
        
        propagation_links = []
        
        # Find molecular targets
        molecular_targets = self._find_targets_at_scale(ingredient_atom, BiologicalScale.MOLECULAR)
        
        for mol_target in molecular_targets:
            # Trace propagation path through scales
            current_scale_idx = 0
            current_targets = [mol_target]
            
            while current_scale_idx < len(self.scale_hierarchy) - 1:
                next_scale = self.scale_hierarchy[current_scale_idx + 1]
                next_targets = self._find_downstream_targets(current_targets, next_scale)
                
                if next_targets:
                    # Create propagation link
                    propagation_efficiency = self._estimate_propagation_efficiency(
                        current_targets, next_targets
                    )
                    
                    propagation_link = EffectPropagationLink(
                        source_action=ActionAtom(f"{mol_target.target_name}_ACTIVATION", "activation", 0.8),
                        target_scale=next_scale,
                        propagation_efficiency=propagation_efficiency,
                        intermediate_targets=current_targets
                    )
                    propagation_links.append(propagation_link)
                    
                    current_targets = next_targets
                current_scale_idx += 1
        
        return propagation_links
    
    def _find_targets_at_scale(self, ingredient: IngredientAtom, scale: BiologicalScale) -> List[BiologicalTargetAtom]:
        """Find biological targets at specific scale for ingredient"""
        targets = []
        
        for link in self.ontology.links.values():
            if isinstance(link, BindingLink):
                if ingredient in link.ingredients and link.target.biological_scale == scale:
                    targets.append(link.target)
        
        return targets
    
    def _find_downstream_targets(self, current_targets: List[BiologicalTargetAtom], 
                                next_scale: BiologicalScale) -> List[BiologicalTargetAtom]:
        """Find targets at next scale that are downstream of current targets"""
        # This would use biological pathway databases (e.g., KEGG, Reactome)
        # For demonstration, simplified heuristic approach
        
        downstream_targets = []
        
        for target in current_targets:
            # Look up known downstream pathways
            if target.target_name == 'RETINOIC_ACID_RECEPTOR' and next_scale == BiologicalScale.CELLULAR:
                downstream_targets.append(
                    BiologicalTargetAtom('COLLAGEN_GENE_EXPRESSION', 'pathway', BiologicalScale.CELLULAR)
                )
            elif target.target_name == 'COLLAGEN_GENE_EXPRESSION' and next_scale == BiologicalScale.TISSUE:
                downstream_targets.append(
                    BiologicalTargetAtom('COLLAGEN_MATRIX', 'structure', BiologicalScale.TISSUE)
                )
        
        return downstream_targets
    
    def _estimate_propagation_efficiency(self, source_targets: List[BiologicalTargetAtom], 
                                       target_targets: List[BiologicalTargetAtom]) -> float:
        """Estimate efficiency of effect propagation between scales"""
        # Simplified model - in practice would use systems biology models
        base_efficiency = 0.7
        
        # Efficiency decreases with number of intermediate steps
        step_penalty = 0.1 * len(source_targets)
        
        return max(0.1, base_efficiency - step_penalty)
```

## 4. Hypergraph Query and Reasoning

### 4.1 Complex Relationship Queries

```python
class HypergraphQueryEngine:
    """Query engine for complex hypergraph relationships"""
    
    def __init__(self, ontology: HypergraphOntologyBuilder):
        self.ontology = ontology
    
    def find_synergistic_combinations(self, target_effect: str, 
                                    max_ingredients: int = 3) -> List[Tuple]:
        """Find ingredient combinations with synergistic effects for target"""
        
        synergistic_combinations = []
        
        # Find all ingredients that affect the target
        relevant_ingredients = self._find_ingredients_affecting_target(target_effect)
        
        # Check all combinations up to max_ingredients
        from itertools import combinations
        
        for r in range(2, max_ingredients + 1):
            for ingredient_combo in combinations(relevant_ingredients, r):
                # Check if combination has synergistic interaction
                synergy_links = self._find_synergy_links(ingredient_combo, target_effect)
                
                if synergy_links:
                    total_synergy = sum(link.interaction_strength for link in synergy_links)
                    synergistic_combinations.append((ingredient_combo, total_synergy))
        
        # Sort by synergy strength
        synergistic_combinations.sort(key=lambda x: x[1], reverse=True)
        
        return synergistic_combinations
    
    def trace_effect_pathway(self, ingredient: IngredientAtom, 
                           target_outcome: str) -> List[EffectPropagationLink]:
        """Trace pathway from ingredient to target outcome across scales"""
        
        pathway = []
        
        # Start from molecular targets
        molecular_targets = self._find_targets_at_scale(ingredient, BiologicalScale.MOLECULAR)
        
        for mol_target in molecular_targets:
            # Use breadth-first search to find pathway to target outcome
            current_path = [mol_target]
            visited = {mol_target}
            
            while current_path:
                current_target = current_path[-1]
                
                # Check if we've reached the target outcome
                if self._target_produces_outcome(current_target, target_outcome):
                    # Convert path to EffectPropagationLinks
                    for i in range(len(current_path) - 1):
                        propagation_link = self._create_propagation_link(
                            current_path[i], current_path[i+1]
                        )
                        pathway.append(propagation_link)
                    break
                
                # Find downstream targets
                downstream = self._find_directly_connected_targets(current_target)
                for target in downstream:
                    if target not in visited:
                        visited.add(target)
                        current_path.append(target)
        
        return pathway
    
    def validate_formulation_constraints(self, formulation: Dict[str, float]) -> Dict:
        """Validate formulation against all constraints in hypergraph"""
        
        validation_results = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'constraint_satisfaction_score': 1.0
        }
        
        total_constraints = 0
        satisfied_constraints = 0
        
        # Check all constraint atoms
        for atom_id, atom in self.ontology.atoms.items():
            if isinstance(atom, ConstraintAtom):
                total_constraints += 1
                
                violation = self._check_constraint_violation(atom, formulation)
                
                if violation:
                    if atom.enforcement_level == 'hard':
                        validation_results['violations'].append(violation)
                        validation_results['valid'] = False
                    else:
                        validation_results['warnings'].append(violation)
                        satisfied_constraints += 0.5  # Partial satisfaction for soft constraints
                else:
                    satisfied_constraints += 1
        
        if total_constraints > 0:
            validation_results['constraint_satisfaction_score'] = satisfied_constraints / total_constraints
        
        return validation_results
    
    def _find_ingredients_affecting_target(self, target_effect: str) -> List[IngredientAtom]:
        """Find all ingredients that affect a specific target"""
        affecting_ingredients = []
        
        for link in self.ontology.links.values():
            if isinstance(link, BindingLink):
                if target_effect.lower() in link.target.target_name.lower():
                    affecting_ingredients.extend(link.ingredients)
        
        return list(set(affecting_ingredients))  # Remove duplicates
    
    def _find_synergy_links(self, ingredients: Tuple[IngredientAtom], 
                          target_effect: str) -> List[InteractionLink]:
        """Find synergistic interaction links between ingredients for target effect"""
        synergy_links = []
        
        for link in self.ontology.links.values():
            if isinstance(link, InteractionLink):
                if (link.interaction_type == 'synergistic' and 
                    set(ingredients).issubset(set(link.ingredients)) and
                    target_effect in link.affected_properties):
                    synergy_links.append(link)
        
        return synergy_links
    
    def _check_constraint_violation(self, constraint: ConstraintAtom, 
                                  formulation: Dict[str, float]) -> Optional[str]:
        """Check if constraint is violated by formulation"""
        
        if constraint.constraint_type == 'CONCENTRATION_LIMIT':
            ingredient = constraint.parameters['ingredient']
            max_conc = constraint.parameters['max_concentration']
            current_conc = formulation.get(ingredient, 0.0)
            
            if current_conc > max_conc:
                return f"{ingredient} concentration {current_conc}% exceeds limit {max_conc}%"
        
        elif constraint.constraint_type == 'PH_RANGE':
            # Would need to calculate formulation pH based on ingredients
            # Simplified check for demonstration
            pass
        
        return None
```

### 4.2 Probabilistic Reasoning Integration

```python
class ProbabilisticHypergraphReasoner:
    """PLN-inspired probabilistic reasoning over hypergraph"""
    
    def __init__(self, query_engine: HypergraphQueryEngine):
        self.query_engine = query_engine
    
    def infer_formulation_success_probability(self, formulation: Dict[str, float], 
                                            target_properties: Dict[str, float]) -> TruthValue:
        """Infer probability of formulation success using PLN-style reasoning"""
        
        # Deduction: Ingredient properties → Formulation properties → Success probability
        
        property_probabilities = {}
        
        for property_name, target_value in target_properties.items():
            # Find ingredients that contribute to this property
            contributing_ingredients = self._find_property_contributors(property_name)
            
            # Calculate individual contributions
            individual_contributions = []
            for ingredient_name, concentration in formulation.items():
                if ingredient_name in contributing_ingredients:
                    contribution = self._calculate_ingredient_contribution(
                        ingredient_name, property_name, concentration
                    )
                    individual_contributions.append(contribution)
            
            # Apply interaction effects
            combined_effect = self._apply_interaction_effects(
                individual_contributions, formulation, property_name
            )
            
            # Calculate probability of achieving target value
            property_prob = self._calculate_target_achievement_probability(
                combined_effect, target_value
            )
            
            property_probabilities[property_name] = property_prob
        
        # Combine property probabilities (assuming independence for simplicity)
        overall_success_prob = 1.0
        total_evidence = 0
        
        for prop_name, prob in property_probabilities.items():
            overall_success_prob *= prob.strength
            total_evidence += prob.count
        
        return TruthValue(
            strength=overall_success_prob ** (1/len(property_probabilities)),  # Geometric mean
            count=total_evidence
        )
    
    def _find_property_contributors(self, property_name: str) -> List[str]:
        """Find ingredients that contribute to a specific property"""
        contributors = []
        
        for atom in self.query_engine.ontology.atoms.values():
            if isinstance(atom, IngredientAtom):
                if property_name in atom.function_category:
                    contributors.append(atom.inci_name)
        
        return contributors
    
    def _calculate_ingredient_contribution(self, ingredient_name: str, 
                                         property_name: str, 
                                         concentration: float) -> TruthValue:
        """Calculate individual ingredient contribution to property"""
        
        # Find ingredient atom
        ingredient_atom = None
        for atom in self.query_engine.ontology.atoms.values():
            if isinstance(atom, IngredientAtom) and atom.inci_name == ingredient_name:
                ingredient_atom = atom
                break
        
        if not ingredient_atom:
            return TruthValue(0.0, 0)
        
        # Simple dose-response model (could be replaced with more sophisticated models)
        base_efficacy = 0.5  # Base efficacy for the property
        max_concentration = 10.0  # Typical maximum concentration
        
        # Sigmoid dose-response
        efficacy = base_efficacy * (concentration / (concentration + max_concentration))
        
        return TruthValue(efficacy, ingredient_atom.truth_value.count)
    
    def _apply_interaction_effects(self, individual_contributions: List[TruthValue], 
                                 formulation: Dict[str, float], 
                                 property_name: str) -> TruthValue:
        """Apply ingredient interaction effects to individual contributions"""
        
        if len(individual_contributions) <= 1:
            return individual_contributions[0] if individual_contributions else TruthValue(0.0, 0)
        
        # Start with additive assumption
        combined_strength = sum(contrib.strength for contrib in individual_contributions)
        combined_count = sum(contrib.count for contrib in individual_contributions)
        
        # Find and apply interaction effects
        interaction_multiplier = 1.0
        
        for link in self.query_engine.ontology.links.values():
            if isinstance(link, InteractionLink):
                if property_name in link.affected_properties:
                    # Check if formulation contains interacting ingredients
                    formulation_ingredients = set(formulation.keys())
                    link_ingredients = set(atom.inci_name for atom in link.ingredients)
                    
                    if link_ingredients.issubset(formulation_ingredients):
                        if link.interaction_type == 'synergistic':
                            interaction_multiplier *= (1 + link.interaction_strength)
                        elif link.interaction_type == 'antagonistic':
                            interaction_multiplier *= (1 - link.interaction_strength)
        
        return TruthValue(
            strength=min(1.0, combined_strength * interaction_multiplier),
            count=combined_count
        )
    
    def _calculate_target_achievement_probability(self, achieved_effect: TruthValue, 
                                                target_value: float) -> TruthValue:
        """Calculate probability of achieving target value given achieved effect"""
        
        # Gaussian probability model around target
        difference = abs(achieved_effect.strength - target_value)
        variance = 0.1  # Assumed variance in achievement
        
        achievement_prob = math.exp(-0.5 * (difference / variance) ** 2)
        
        return TruthValue(achievement_prob, achieved_effect.count)

class TruthValue:
    """PLN-style truth value with strength and confidence"""
    
    def __init__(self, strength: float, count: float):
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0,1]
        self.count = max(0.0, count)
    
    @property
    def confidence(self) -> float:
        """Calculate confidence from count"""
        return self.count / (self.count + 1.0)
    
    def __str__(self):
        return f"TV({self.strength:.3f}, {self.confidence:.3f})"
```

## 5. Integration with Optimization System

### 5.1 Hypergraph-Guided Search Space Reduction

The hypergraph ontology enables intelligent search space reduction by:

1. **Constraint Propagation**: Follow constraint links to eliminate invalid ingredient combinations
2. **Interaction Filtering**: Use interaction links to identify promising synergistic combinations
3. **Scale-Aware Optimization**: Optimize effects at appropriate biological scales
4. **Probabilistic Pruning**: Use truth values to focus on high-confidence regions

### 5.2 Implementation Example

```python
class HypergraphGuidedOptimizer:
    """Optimizer that uses hypergraph ontology for intelligent search"""
    
    def __init__(self, ontology: HypergraphOntologyBuilder):
        self.ontology = ontology
        self.query_engine = HypergraphQueryEngine(ontology)
        self.reasoner = ProbabilisticHypergraphReasoner(self.query_engine)
    
    def optimize_formulation(self, target_properties: Dict[str, float], 
                           constraints: List[ConstraintAtom]) -> Dict:
        """Optimize formulation using hypergraph guidance"""
        
        # Step 1: Identify relevant ingredients using hypergraph queries
        relevant_ingredients = self._identify_relevant_ingredients(target_properties)
        
        # Step 2: Find synergistic combinations
        synergistic_combos = self.query_engine.find_synergistic_combinations(
            list(target_properties.keys())[0], max_ingredients=5
        )
        
        # Step 3: Generate initial population biased toward promising combinations
        initial_population = self._generate_biased_population(
            relevant_ingredients, synergistic_combos
        )
        
        # Step 4: Use hypergraph-guided fitness evaluation
        best_formulation = None
        best_fitness = 0.0
        
        for candidate in initial_population:
            fitness = self._hypergraph_fitness_evaluation(candidate, target_properties)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_formulation = candidate
        
        return {
            'best_formulation': best_formulation,
            'best_fitness': best_fitness,
            'optimization_method': 'hypergraph_guided'
        }
    
    def _hypergraph_fitness_evaluation(self, formulation: Dict[str, float], 
                                     target_properties: Dict[str, float]) -> float:
        """Evaluate formulation fitness using hypergraph reasoning"""
        
        # Use probabilistic reasoning to estimate success probability
        success_probability = self.reasoner.infer_formulation_success_probability(
            formulation, target_properties
        )
        
        # Check constraint satisfaction  
        constraint_validation = self.query_engine.validate_formulation_constraints(formulation)
        
        # Combine success probability with constraint satisfaction
        fitness = success_probability.strength * constraint_validation['constraint_satisfaction_score']
        
        return fitness
```

## 6. Conclusion

The hypergraph encoding of ingredient and action ontologies provides a powerful foundation for intelligent cosmeceutical formulation optimization. Key advantages include:

1. **Rich Relationship Modeling**: Capture complex N-ary relationships between ingredients, targets, and effects
2. **Multi-Scale Integration**: Bridge molecular actions to organ-level outcomes
3. **Probabilistic Reasoning**: Handle uncertainty through PLN-inspired truth values  
4. **Constraint Propagation**: Efficiently enforce complex regulatory and safety constraints
5. **Synergy Discovery**: Automatically identify beneficial ingredient combinations
6. **Intelligent Search**: Guide optimization toward promising regions of formulation space

This hypergraph-based approach enables the system to reason about formulations in a biologically-meaningful way, leading to more effective and safer cosmeceutical products.

---

*This documentation describes the theoretical framework and implementation approach for hypergraph ontology encoding in the OpenCog Multiscale Constraint Optimization system. For practical implementation details, see the source code in the examples/python directory.*