#!/usr/bin/env python3
#
# test_inci_advanced.py
#
# Advanced Test Cases for INCI-Based Search Space Pruning and Optimization Accuracy
# Comprehensive validation of INCI-driven formulation optimization with real-world scenarios
#
# Part of the OpenCog Multiscale Constraint Optimization system
# Tests specifically designed to validate the deliverables mentioned in the issue:
# - INCI-based search space reduction algorithms
# - Absolute concentration estimation accuracy
# - Regulatory compliance validation
# - Optimization accuracy across diverse formulation types
# --------------------------------------------------------------

import unittest
import time
import random
import math
from typing import Dict, List, Tuple

# Import our modules
from inci_optimizer import (
    INCISearchSpaceReducer, INCIParser, FormulationConstraint, 
    IngredientInfo, RegionType
)
from multiscale_optimizer import MultiscaleConstraintOptimizer
from attention_allocation import AttentionAllocationManager

class TestINCISearchSpacePruning(unittest.TestCase):
    """Advanced test cases for INCI-based search space pruning accuracy"""
    
    def setUp(self):
        self.reducer = INCISearchSpaceReducer()
        self.optimizer = MultiscaleConstraintOptimizer()
        
    def test_complex_inci_parsing_accuracy(self):
        """Test parsing accuracy for complex real-world INCI lists"""
        
        test_cases = [
            {
                'name': 'Premium Anti-Aging Serum',
                'inci': 'AQUA, GLYCERIN, NIACINAMIDE, SODIUM HYALURONATE, RETINOL, ASCORBYL PALMITATE, TOCOPHEROL, PHENOXYETHANOL, ETHYLHEXYLGLYCERIN',
                'expected_water_range': (55, 70),  # Water should be 55-70%
                'expected_active_count': 4,  # NIACINAMIDE, RETINOL, ASCORBYL PALMITATE, TOCOPHEROL
                'expected_preservative_count': 2  # PHENOXYETHANOL, ETHYLHEXYLGLYCERIN
            },
            {
                'name': 'Sensitive Skin Moisturizer',
                'inci': 'AQUA, CETYL ALCOHOL, GLYCERIN, DIMETHICONE, NIACINAMIDE, CERAMIDE NP, HYALURONIC ACID, ALLANTOIN, CARBOMER, SODIUM HYDROXIDE',
                'expected_water_range': (50, 65),
                'expected_emulsifier_count': 2,  # CETYL ALCOHOL, DIMETHICONE
                'expected_actives': ['NIACINAMIDE', 'CERAMIDE NP', 'HYALURONIC ACID', 'ALLANTOIN']
            },
            {
                'name': 'Vitamin C Brightening Serum',
                'inci': 'AQUA, ASCORBIC ACID, PROPYLENE GLYCOL, GLYCERIN, SODIUM ASCORBYL PHOSPHATE, KOJIC ACID, ARBUTIN, FERULIC ACID, SODIUM HYDROXIDE, PHENOXYETHANOL',
                'expected_water_range': (60, 75),
                'expected_vitamin_c_forms': 2,  # ASCORBIC ACID, SODIUM ASCORBYL PHOSPHATE
                'expected_brightening_agents': 3  # KOJIC ACID, ARBUTIN, FERULIC ACID
            }
        ]
        
        for case in test_cases:
            with self.subTest(formulation=case['name']):
                parser = INCIParser()
                parsed = parser.parse_inci_list(case['inci'])
                
                # Test water concentration estimation
                if 'expected_water_range' in case:
                    water_conc = next(
                        (conc for name, conc in parsed if name == 'AQUA'), 
                        None
                    )
                    self.assertIsNotNone(water_conc, f"Water not found in {case['name']}")
                    self.assertTrue(
                        case['expected_water_range'][0] <= water_conc <= case['expected_water_range'][1],
                        f"Water concentration {water_conc}% outside expected range {case['expected_water_range']}"
                    )
                
                # Test active ingredient identification
                if 'expected_actives' in case:
                    parsed_names = [name for name, _ in parsed]
                    for active in case['expected_actives']:
                        self.assertIn(
                            active, parsed_names,
                            f"Expected active {active} not found in parsed ingredients"
                        )
                
                # Verify total adds to ~100%
                total_concentration = sum(conc for _, conc in parsed)
                self.assertTrue(
                    95 <= total_concentration <= 105,
                    f"Total concentration {total_concentration}% should be ~100%"
                )
    
    def test_regulatory_compliance_validation(self):
        """Test regulatory compliance validation across different regions"""
        
        test_formulations = [
            {
                'name': 'EU Compliant Sunscreen',
                'inci': 'AQUA, ZINC OXIDE, TITANIUM DIOXIDE, OCTYL METHOXYCINNAMATE, GLYCERIN, CETYL ALCOHOL',
                'region': RegionType.EU,
                'should_comply': True,
                'expected_issues': []
            },
            {
                'name': 'High Retinol Serum (EU Non-compliant)',
                'inci': 'AQUA, RETINOL, GLYCERIN, TOCOPHEROL',  # Retinol too high by INCI position
                'region': RegionType.EU,
                'should_comply': False,
                'expected_issues': ['RETINOL concentration exceeds EU limits']
            },
            {
                'name': 'Hydroquinone Cream (FDA Only)',
                'inci': 'AQUA, HYDROQUINONE, GLYCERIN, CETYL ALCOHOL',
                'region': RegionType.FDA,
                'should_comply': True,
                'expected_issues': []
            }
        ]
        
        for formulation in test_formulations:
            with self.subTest(formulation=formulation['name']):
                compliance_result = self.reducer.check_regulatory_compliance(
                    formulation['inci'], 
                    formulation['region']
                )
                
                if formulation['should_comply']:
                    self.assertTrue(
                        compliance_result['compliant'],
                        f"{formulation['name']} should be compliant but isn't"
                    )
                else:
                    self.assertFalse(
                        compliance_result['compliant'],
                        f"{formulation['name']} should not be compliant but is"
                    )
                    
                # Check that expected issues are detected
                for expected_issue in formulation.get('expected_issues', []):
                    issue_found = any(
                        expected_issue.lower() in issue.lower() 
                        for issue in compliance_result.get('issues', [])
                    )
                    self.assertTrue(
                        issue_found,
                        f"Expected issue '{expected_issue}' not found in compliance check"
                    )
    
    def test_search_space_reduction_accuracy(self):
        """Test accuracy of search space reduction for different formulation types"""
        
        formulation_types = [
            {
                'type': 'Anti-Aging Serum',
                'base_inci': 'AQUA, GLYCERIN, NIACINAMIDE, RETINOL, HYALURONIC ACID',
                'expected_categories': ['antioxidants', 'moisturizers', 'anti-aging', 'preservatives'],
                'min_reduction_factor': 100,  # Should achieve at least 100x reduction
                'max_ingredients': 12
            },
            {
                'type': 'Sunscreen',
                'base_inci': 'AQUA, ZINC OXIDE, TITANIUM DIOXIDE, OCTYL METHOXYCINNAMATE',
                'expected_categories': ['uv_filters', 'emulsifiers', 'stabilizers'],
                'min_reduction_factor': 50,
                'max_ingredients': 15
            },
            {
                'type': 'Sensitive Skin Moisturizer',
                'base_inci': 'AQUA, GLYCERIN, CERAMIDE NP, ALLANTOIN, PANTHENOL',
                'expected_categories': ['moisturizers', 'skin_barrier', 'soothing'],
                'min_reduction_factor': 200,
                'max_ingredients': 10
            }
        ]
        
        for formulation in formulation_types:
            with self.subTest(formulation_type=formulation['type']):
                constraints = [
                    FormulationConstraint('AQUA', 40, 70, required=True),
                    FormulationConstraint('GLYCERIN', 5, 20)
                ]
                
                reduction_result = self.reducer.reduce_search_space(
                    formulation['base_inci'],
                    constraints,
                    formulation['max_ingredients']
                )
                
                # Check reduction factor
                self.assertGreaterEqual(
                    reduction_result['reduction_factor'],
                    formulation['min_reduction_factor'],
                    f"Reduction factor {reduction_result['reduction_factor']} below minimum {formulation['min_reduction_factor']}"
                )
                
                # Check that viable ingredients are appropriate for formulation type
                viable_ingredients = reduction_result['viable_ingredients']
                self.assertGreater(
                    len(viable_ingredients), 0,
                    "No viable ingredients found"
                )
                
                # Verify processing time is acceptable
                self.assertLess(
                    reduction_result['processing_time_ms'], 1.0,
                    "Search space reduction taking too long"
                )
    
    def test_concentration_estimation_accuracy(self):
        """Test accuracy of absolute concentration estimation from INCI ordering"""
        
        known_formulations = [
            {
                'name': 'Simple Moisturizer',
                'inci': 'AQUA, GLYCERIN, CETYL ALCOHOL, DIMETHICONE, PHENOXYETHANOL',
                'known_concentrations': {
                    'AQUA': 65.0,
                    'GLYCERIN': 15.0,
                    'CETYL ALCOHOL': 8.0,
                    'DIMETHICONE': 10.0,
                    'PHENOXYETHANOL': 1.0
                },
                'tolerance': 0.15  # 15% tolerance for estimation
            },
            {
                'name': 'Serum Formulation',
                'inci': 'AQUA, PROPYLENE GLYCOL, NIACINAMIDE, SODIUM HYALURONATE, CARBOMER, PHENOXYETHANOL',
                'known_concentrations': {
                    'AQUA': 70.0,
                    'PROPYLENE GLYCOL': 12.0,
                    'NIACINAMIDE': 5.0,
                    'SODIUM HYALURONATE': 2.0,
                    'CARBOMER': 0.5,
                    'PHENOXYETHANOL': 0.5
                },
                'tolerance': 0.20
            }
        ]
        
        for formulation in known_formulations:
            with self.subTest(formulation=formulation['name']):
                estimated_concentrations = self.reducer.estimate_absolute_concentrations(
                    formulation['inci'],
                    total_volume=100.0
                )
                
                for ingredient, known_conc in formulation['known_concentrations'].items():
                    estimated_conc = estimated_concentrations.get(ingredient, 0)
                    
                    relative_error = abs(estimated_conc - known_conc) / known_conc
                    self.assertLessEqual(
                        relative_error, formulation['tolerance'],
                        f"Concentration estimation for {ingredient}: "
                        f"estimated {estimated_conc}%, known {known_conc}%, "
                        f"error {relative_error:.1%} > tolerance {formulation['tolerance']:.1%}"
                    )
    
    def test_optimization_accuracy_with_inci_constraints(self):
        """Test optimization accuracy when constrained by INCI-derived information"""
        
        optimization_scenarios = [
            {
                'name': 'Match Commercial Anti-Aging Serum',
                'target_inci': 'AQUA, GLYCERIN, NIACINAMIDE, RETINOL, ASCORBIC ACID, TOCOPHEROL',
                'target_properties': {
                    'anti_aging_efficacy': 0.8,
                    'safety_score': 0.9,
                    'cost_efficiency': 0.6,
                    'stability': 0.7
                },
                'constraints': [
                    FormulationConstraint('AQUA', 50, 70, required=True),
                    FormulationConstraint('NIACINAMIDE', 2, 10),
                    FormulationConstraint('RETINOL', 0.1, 2.0)
                ],
                'success_threshold': 0.7  # Minimum fitness score for success
            },
            {
                'name': 'Optimize Sensitive Skin Formula',
                'target_inci': 'AQUA, GLYCERIN, CERAMIDE NP, ALLANTOIN, PANTHENOL, PHENOXYETHANOL',
                'target_properties': {
                    'gentleness': 0.9,
                    'moisturizing': 0.8,
                    'safety_score': 0.95,
                    'cost_efficiency': 0.5
                },
                'constraints': [
                    FormulationConstraint('AQUA', 60, 75, required=True),
                    FormulationConstraint('CERAMIDE NP', 1, 5),
                    FormulationConstraint('ALLANTOIN', 0.1, 1.0)
                ],
                'success_threshold': 0.75
            }
        ]
        
        for scenario in optimization_scenarios:
            with self.subTest(scenario=scenario['name']):
                start_time = time.time()
                
                optimization_result = self.optimizer.optimize_formulation(
                    target_profile=scenario['target_properties'],
                    constraints=scenario['constraints'],
                    base_ingredients=None
                )
                
                optimization_time = time.time() - start_time
                
                # Check optimization succeeded
                self.assertGreater(
                    optimization_result['best_fitness'],
                    scenario['success_threshold'],
                    f"Optimization fitness {optimization_result['best_fitness']} "
                    f"below success threshold {scenario['success_threshold']}"
                )
                
                # Check optimization time is reasonable
                self.assertLess(
                    optimization_time, 60.0,
                    f"Optimization took {optimization_time}s, should be <60s"
                )
                
                # Verify that the solution respects INCI-derived constraints
                best_formulation = optimization_result['best_candidate']
                for constraint in scenario['constraints']:
                    ingredient_conc = best_formulation.ingredients.get(constraint.ingredient, 0)
                    
                    if constraint.required:
                        self.assertGreater(
                            ingredient_conc, 0,
                            f"Required ingredient {constraint.ingredient} missing from solution"
                        )
                    
                    self.assertTrue(
                        constraint.min_concentration <= ingredient_conc <= constraint.max_concentration,
                        f"Ingredient {constraint.ingredient} concentration {ingredient_conc}% "
                        f"outside constraint range [{constraint.min_concentration}, {constraint.max_concentration}]"
                    )
    
    def test_edge_case_inci_handling(self):
        """Test handling of edge cases in INCI parsing and optimization"""
        
        edge_cases = [
            {
                'name': 'Single Ingredient',
                'inci': 'AQUA',
                'should_parse': True,
                'expected_concentration': 100.0
            },
            {
                'name': 'Very Long INCI List',
                'inci': ', '.join([f'INGREDIENT_{i}' for i in range(50)]),
                'should_parse': True,
                'min_ingredients': 40  # Should handle at least 40 ingredients
            },
            {
                'name': 'Special Characters in INCI',
                'inci': 'AQUA, PEG-100 STEARATE, C12-15 ALKYL BENZOATE, DIMETHICONE/VINYL DIMETHICONE CROSSPOLYMER',
                'should_parse': True,
                'expected_special_chars': True
            },
            {
                'name': 'Empty INCI',
                'inci': '',
                'should_parse': False,
                'expected_error': 'Empty INCI list'
            },
            {
                'name': 'Invalid Characters',
                'inci': 'AQUA, GLYCERIN@#$%, NIACINAMIDE',
                'should_parse': True,  # Should handle gracefully
                'expected_cleanup': True
            }
        ]
        
        parser = INCIParser()
        
        for case in edge_cases:
            with self.subTest(case=case['name']):
                if case['should_parse']:
                    try:
                        parsed = parser.parse_inci_list(case['inci'])
                        
                        if 'expected_concentration' in case:
                            self.assertEqual(
                                len(parsed), 1,
                                "Single ingredient should result in one parsed item"
                            )
                            self.assertEqual(
                                parsed[0][1], case['expected_concentration'],
                                f"Single ingredient concentration should be {case['expected_concentration']}%"
                            )
                        
                        if 'min_ingredients' in case:
                            self.assertGreaterEqual(
                                len(parsed), case['min_ingredients'],
                                f"Should parse at least {case['min_ingredients']} ingredients"
                            )
                        
                        if 'expected_special_chars' in case:
                            special_char_found = any(
                                any(char in name for char in ['-', '/', '(', ')'])
                                for name, _ in parsed
                            )
                            self.assertTrue(
                                special_char_found,
                                "Should preserve special characters in ingredient names"
                            )
                            
                    except Exception as e:
                        self.fail(f"Parsing failed for {case['name']}: {e}")
                        
                else:
                    with self.assertRaises(Exception):
                        parser.parse_inci_list(case['inci'])
    
    def test_performance_benchmarks(self):
        """Test that INCI processing meets performance requirements"""
        
        # Generate test data of varying complexity
        test_datasets = [
            {
                'name': 'Simple Formulations',
                'formulations': [
                    'AQUA, GLYCERIN, PHENOXYETHANOL',
                    'AQUA, DIMETHICONE, CYCLOPENTASILOXANE',
                    'AQUA, CETYL ALCOHOL, GLYCERIN'
                ] * 100,  # 300 simple formulations
                'max_time_per_parse': 0.01  # 0.01ms per parse
            },
            {
                'name': 'Complex Formulations',
                'formulations': [
                    'AQUA, GLYCERIN, NIACINAMIDE, SODIUM HYALURONATE, RETINOL, ASCORBIC ACID, TOCOPHEROL, DIMETHICONE, CETYL ALCOHOL, STEARIC ACID, GLYCERYL STEARATE, PEG-100 STEARATE, CARBOMER, SODIUM HYDROXIDE, PHENOXYETHANOL, ETHYLHEXYLGLYCERIN'
                ] * 50,  # 50 complex formulations
                'max_time_per_parse': 0.05  # 0.05ms per parse
            }
        ]
        
        parser = INCIParser()
        
        for dataset in test_datasets:
            with self.subTest(dataset=dataset['name']):
                total_time = 0
                successful_parses = 0
                
                for inci_string in dataset['formulations']:
                    start_time = time.perf_counter()
                    
                    try:
                        parsed = parser.parse_inci_list(inci_string)
                        self.assertGreater(len(parsed), 0, "Parse result should not be empty")
                        successful_parses += 1
                    except Exception as e:
                        self.fail(f"Parse failed for INCI: {inci_string[:50]}... Error: {e}")
                    
                    end_time = time.perf_counter()
                    parse_time = (end_time - start_time) * 1000  # Convert to ms
                    total_time += parse_time
                    
                    self.assertLess(
                        parse_time, dataset['max_time_per_parse'],
                        f"Parse time {parse_time:.3f}ms exceeds limit {dataset['max_time_per_parse']}ms"
                    )
                
                # Overall performance metrics
                avg_time = total_time / len(dataset['formulations'])
                success_rate = successful_parses / len(dataset['formulations'])
                
                self.assertGreaterEqual(
                    success_rate, 0.99,
                    f"Success rate {success_rate:.1%} below 99%"
                )
                
                print(f"\n{dataset['name']} Performance:")
                print(f"  Average parse time: {avg_time:.3f}ms")
                print(f"  Success rate: {success_rate:.1%}")
                print(f"  Total formulations: {len(dataset['formulations'])}")

class TestINCIIntegrationAccuracy(unittest.TestCase):
    """Test integration accuracy between INCI processing and optimization"""
    
    def setUp(self):
        self.reducer = INCISearchSpaceReducer()
        self.optimizer = MultiscaleConstraintOptimizer(inci_reducer=self.reducer)
        self.attention_manager = AttentionAllocationManager()
    
    def test_end_to_end_optimization_accuracy(self):
        """Test end-to-end optimization accuracy starting from INCI analysis"""
        
        commercial_products = [
            {
                'name': 'The Ordinary Niacinamide 10% + Zinc 1%',
                'inci': 'AQUA, NIACINAMIDE, PENTYLENE GLYCOL, ZINC PCA, DIMETHYL ISOSORBIDE, TAMARINDUS INDICA SEED GUM, XANTHAN GUM, ISOCETETH-20, ETHOXYDIGLYCOL, PHENOXYETHANOL, CHLORPHENESIN',
                'expected_niacinamide': 10.0,
                'expected_zinc': 1.0,
                'tolerance': 0.2  # 20% tolerance for reverse engineering
            },
            {
                'name': 'CeraVe Daily Moisturizing Lotion',
                'inci': 'AQUA, GLYCERIN, CAPRYLIC/CAPRIC TRIGLYCERIDE, BEHENTRIMONIUM METHOSULFATE, CETEARYL ALCOHOL, CERAMIDE NP, CERAMIDE AP, CERAMIDE EOP, CARBOMER, DIMETHICONE, PHENOXYETHANOL',
                'expected_water': 70.0,
                'expected_glycerin': 15.0,
                'tolerance': 0.25
            }
        ]
        
        for product in commercial_products:
            with self.subTest(product=product['name']):
                # Step 1: Analyze existing formulation
                analysis_result = self.reducer.reduce_search_space(
                    product['inci'],
                    constraints=[],
                    max_ingredients=20
                )
                
                parsed_ingredients = analysis_result['target_ingredients']
                
                # Step 2: Verify key ingredient concentrations
                for ingredient_name, expected_conc in product.items():
                    if ingredient_name.startswith('expected_'):
                        ingredient = ingredient_name.replace('expected_', '').upper()
                        
                        found_conc = next(
                            (conc for name, conc in parsed_ingredients if name == ingredient),
                            None
                        )
                        
                        if found_conc is not None:
                            relative_error = abs(found_conc - expected_conc) / expected_conc
                            self.assertLessEqual(
                                relative_error, product['tolerance'],
                                f"Concentration error for {ingredient}: "
                                f"found {found_conc}%, expected {expected_conc}%, "
                                f"error {relative_error:.1%}"
                            )
                
                # Step 3: Optimize similar formulation
                target_profile = {
                    'efficacy': 0.8,
                    'safety': 0.9,
                    'cost': 0.6,
                    'stability': 0.7
                }
                
                optimization_result = self.optimizer.optimize_formulation(
                    target_profile=target_profile,
                    constraints=[],
                    base_ingredients=[name for name, _ in parsed_ingredients[:5]]
                )
                
                # Step 4: Validate optimization result
                self.assertGreater(
                    optimization_result['best_fitness'], 0.5,
                    f"Optimization fitness too low for {product['name']}"
                )
                
                # Verify the optimized formulation is chemically plausible
                best_formulation = optimization_result['best_candidate']
                total_concentration = sum(best_formulation.ingredients.values())
                
                self.assertTrue(
                    95 <= total_concentration <= 105,
                    f"Total concentration {total_concentration}% should be ~100%"
                )

if __name__ == '__main__':
    print("=" * 80)
    print("Advanced INCI-Based Search Space Pruning and Optimization Accuracy Tests")
    print("=" * 80)
    
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)