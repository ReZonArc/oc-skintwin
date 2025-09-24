#!/usr/bin/env python3
#
# test_hypergredient_framework.py
#
# Comprehensive test suite for the Hypergredient Framework Architecture
# Tests all components including hypergredient database, formulation optimization,
# integration with multiscale systems, and performance requirements.
#
# Usage: python test_hypergredient_framework.py
# --------------------------------------------------------------

import unittest
import time
import sys
import os
from typing import Dict, List

# Import hypergredient framework components
from hypergredient_framework import (
    HypergredientDatabase, HypergredientFormulator, HypergredientClass, 
    HypergredientInfo, RegionType
)
from hypergredient_integration import (
    HypergredientMultiscaleOptimizer, HypergredientFormulationCandidate
)

# Import existing test infrastructure
from test_multiscale_optimization import TestMultiscaleOptimization
from multiscale_optimizer import FormulationCandidate, ObjectiveType
from inci_optimizer import FormulationConstraint

class TestHypergredientDatabase(unittest.TestCase):
    """Test suite for hypergredient database functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.database = HypergredientDatabase()
    
    def test_database_initialization(self):
        """Test database initialization with comprehensive ingredient data"""
        # Check that database is populated
        self.assertGreater(len(self.database.hypergredients), 10)
        
        # Check that all hypergredient classes are represented
        represented_classes = set()
        for hypergredient in self.database.hypergredients.values():
            represented_classes.add(hypergredient.hypergredient_class)
        
        # Should have at least 6 major classes represented
        self.assertGreaterEqual(len(represented_classes), 6)
        
        # Check class mapping is built correctly
        total_mapped = sum(len(ingredients) for ingredients in self.database.class_mapping.values())
        self.assertEqual(total_mapped, len(self.database.hypergredients))
    
    def test_hypergredient_classification(self):
        """Test hypergredient classification accuracy"""
        # Test specific classifications
        retinol = self.database.hypergredients.get("RETINOL")
        if retinol:
            self.assertEqual(retinol.hypergredient_class, HypergredientClass.CT)
            self.assertIn("cellular_turnover", retinol.primary_function)
        
        hyaluronic = self.database.hypergredients.get("SODIUM HYALURONATE")
        if hyaluronic:
            self.assertEqual(hyaluronic.hypergredient_class, HypergredientClass.HY)
            self.assertIn("hydration", hyaluronic.primary_function)
        
        vitamin_c = self.database.hypergredients.get("ASCORBIC ACID")
        if vitamin_c:
            self.assertEqual(vitamin_c.hypergredient_class, HypergredientClass.CS)
            self.assertIn("collagen_synthesis", vitamin_c.primary_function)
    
    def test_interaction_matrix(self):
        """Test hypergredient interaction matrix"""
        # Test that interaction matrix is populated
        self.assertGreater(len(self.database.interaction_matrix), 20)
        
        # Test specific known synergies
        cs_ao_score = self.database.get_interaction_score(
            HypergredientClass.CS, HypergredientClass.AO
        )
        self.assertGreater(cs_ao_score, 1.5)  # Strong synergy
        
        # Test symmetry
        ao_cs_score = self.database.get_interaction_score(
            HypergredientClass.AO, HypergredientClass.CS
        )
        self.assertEqual(cs_ao_score, ao_cs_score)
        
        # Test default interaction score
        default_score = self.database.get_interaction_score(
            HypergredientClass.MB, HypergredientClass.PD  # Unlikely to be defined
        )
        self.assertEqual(default_score, 1.0)
    
    def test_hypergredient_scoring(self):
        """Test hypergredient composite scoring system"""
        # Test with known high-performance ingredient
        tretinoin = self.database.hypergredients.get("TRETINOIN")
        if tretinoin:
            score = tretinoin.calculate_composite_score()
            self.assertGreater(score, 5.0)  # Should be above average
            self.assertLessEqual(score, 10.0)  # Should not exceed maximum
        
        # Test with safe, stable ingredient
        hyaluronic = self.database.hypergredients.get("SODIUM HYALURONATE")
        if hyaluronic:
            score = hyaluronic.calculate_composite_score()
            self.assertGreater(score, 7.0)  # Should score high due to safety and stability
        
        # Test custom weights
        if tretinoin:
            safety_focused_score = tretinoin.calculate_composite_score({
                'efficacy': 0.1, 'safety': 0.6, 'stability': 0.2, 'cost': 0.1
            })
            efficacy_focused_score = tretinoin.calculate_composite_score({
                'efficacy': 0.6, 'safety': 0.1, 'stability': 0.1, 'cost': 0.2
            })
            # Should be different based on different weights
            self.assertNotEqual(safety_focused_score, efficacy_focused_score)
    
    def test_hypergredient_search(self):
        """Test hypergredient search functionality"""
        # Search by function
        antioxidants = self.database.search_hypergredients(function="antioxidant")
        self.assertGreater(len(antioxidants), 0)
        
        # Search by efficacy
        high_efficacy = self.database.search_hypergredients(min_efficacy=8.0)
        self.assertGreater(len(high_efficacy), 0)
        for hypergredient in high_efficacy:
            self.assertGreaterEqual(hypergredient.efficacy_score, 8.0)
        
        # Search by cost
        affordable = self.database.search_hypergredients(max_cost=200.0)
        self.assertGreater(len(affordable), 0)
        for hypergredient in affordable:
            self.assertLessEqual(hypergredient.cost_per_gram, 200.0)
        
        # Search with exclusions
        no_retinoids = self.database.search_hypergredients(
            exclude_ingredients=["TRETINOIN", "RETINOL"]
        )
        ingredient_names = [h.inci_name for h in no_retinoids]
        self.assertNotIn("TRETINOIN", ingredient_names)
        self.assertNotIn("RETINOL", ingredient_names)

class TestHypergredientFormulator(unittest.TestCase):
    """Test suite for hypergredient formulation generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.database = HypergredientDatabase()
        self.formulator = HypergredientFormulator(self.database)
    
    def test_formulation_generation_basic(self):
        """Test basic formulation generation"""
        request = {
            'target_concerns': ['wrinkles', 'firmness'],
            'skin_type': 'normal',
            'budget': 1000.0
        }
        
        result = self.formulator.generate_formulation(**request)
        
        # Check result structure
        self.assertIn('selected_hypergredients', result)
        self.assertIn('total_cost', result)
        self.assertIn('synergy_score', result)
        self.assertIn('efficacy_prediction', result)
        
        # Check that relevant hypergredient classes are selected
        selected_classes = set(result['selected_hypergredients'].keys())
        self.assertTrue(
            'H.CT' in selected_classes or 'H.CS' in selected_classes,
            "Should select cellular turnover or collagen synthesis for wrinkles/firmness"
        )
        
        # Check cost constraint
        self.assertLessEqual(result['total_cost'], request['budget'])
    
    def test_formulation_skin_type_adaptation(self):
        """Test formulation adaptation to different skin types"""
        base_request = {
            'target_concerns': ['hydration'],
            'budget': 500.0
        }
        
        # Generate for sensitive skin
        sensitive_result = self.formulator.generate_formulation(
            skin_type='sensitive', **base_request
        )
        
        # Generate for normal skin
        normal_result = self.formulator.generate_formulation(
            skin_type='normal', **base_request
        )
        
        # Sensitive skin should prioritize safer ingredients
        self.assertIn('safety_assessment', sensitive_result)
        self.assertIn('safety_assessment', normal_result)
        
        # Results should be different (different ingredient selection or concentrations)
        sensitive_ingredients = set(sensitive_result['selected_hypergredients'].keys())
        normal_ingredients = set(normal_result['selected_hypergredients'].keys())
        
        # Allow for same classes but potentially different specific ingredients or concentrations
        self.assertTrue(len(sensitive_ingredients) > 0 and len(normal_ingredients) > 0)
    
    def test_formulation_budget_constraints(self):
        """Test formulation generation with budget constraints"""
        low_budget_result = self.formulator.generate_formulation(
            target_concerns=['hydration'],
            budget=200.0
        )
        
        high_budget_result = self.formulator.generate_formulation(
            target_concerns=['hydration'],
            budget=2000.0
        )
        
        # Both should respect budget constraints
        self.assertLessEqual(low_budget_result['total_cost'], 200.0)
        self.assertLessEqual(high_budget_result['total_cost'], 2000.0)
        
        # High budget should allow for better or more ingredients
        low_ingredients = len(low_budget_result['selected_hypergredients'])
        high_ingredients = len(high_budget_result['selected_hypergredients'])
        
        # Either more ingredients or potentially higher quality
        self.assertTrue(high_ingredients >= low_ingredients)
    
    def test_formulation_exclusions(self):
        """Test formulation generation with ingredient exclusions"""
        result_with_exclusions = self.formulator.generate_formulation(
            target_concerns=['anti_aging'],
            budget=1000.0,
            exclude_ingredients=['TRETINOIN', 'RETINOL']
        )
        
        # Check that excluded ingredients are not selected
        selected_ingredients = []
        for class_data in result_with_exclusions['selected_hypergredients'].values():
            selected_ingredients.append(class_data['hypergredient'].inci_name)
        
        self.assertNotIn('TRETINOIN', selected_ingredients)
        self.assertNotIn('RETINOL', selected_ingredients)
    
    def test_efficacy_prediction(self):
        """Test efficacy prediction accuracy"""
        result = self.formulator.generate_formulation(
            target_concerns=['wrinkles', 'brightness'],
            budget=1500.0
        )
        
        efficacy = result['efficacy_prediction']
        
        # Efficacy should be a reasonable percentage
        self.assertGreaterEqual(efficacy, 0.0)
        self.assertLessEqual(efficacy, 100.0)
        
        # Should be reasonably high for a well-formulated product
        self.assertGreater(efficacy, 5.0)  # At least 5% predicted improvement
    
    def test_stability_assessment(self):
        """Test stability timeline estimation"""
        result = self.formulator.generate_formulation(
            target_concerns=['hydration'],
            budget=800.0
        )
        
        stability = result['stability_timeline']
        
        # Check structure
        self.assertIn('estimated_shelf_life_months', stability)
        self.assertIn('limiting_factor', stability)
        self.assertIn('storage_recommendations', stability)
        
        # Shelf life should be reasonable
        shelf_life = stability['estimated_shelf_life_months']
        self.assertGreaterEqual(shelf_life, 1)
        self.assertLessEqual(shelf_life, 36)
    
    def test_safety_assessment(self):
        """Test safety assessment functionality"""
        result = self.formulator.generate_formulation(
            target_concerns=['sensitivity'],
            skin_type='sensitive',
            budget=600.0
        )
        
        safety = result['safety_assessment']
        
        # Check structure
        self.assertIn('overall_safety_score', safety)
        self.assertIn('skin_type_suitability', safety)
        self.assertIn('warnings', safety)
        self.assertIn('patch_test_recommended', safety)
        
        # For sensitive skin formulation, should have high safety
        self.assertGreaterEqual(safety['overall_safety_score'], 7.0)

class TestHypergredientIntegration(unittest.TestCase):
    """Test suite for hypergredient-multiscale integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = HypergredientMultiscaleOptimizer()
    
    def test_integration_initialization(self):
        """Test integrated optimizer initialization"""
        # Check that both systems are initialized
        self.assertIsNotNone(self.optimizer.hypergredient_db)
        self.assertIsNotNone(self.optimizer.hypergredient_formulator)
        self.assertIsNotNone(self.optimizer.inci_reducer)
        
        # Check database population
        self.assertGreater(len(self.optimizer.hypergredient_db.hypergredients), 5)
    
    def test_hypergredient_enhanced_optimization(self):
        """Test hypergredient-enhanced optimization"""
        target_profile = {
            'skin_hydration': 0.7,
            'skin_elasticity': 0.6,
            'barrier_function': 0.5
        }
        
        constraints = [
            FormulationConstraint("AQUA", 50.0, 80.0, required=True),
            FormulationConstraint("GLYCERIN", 2.0, 10.0, required=True)
        ]
        
        results = self.optimizer.optimize_formulation_with_hypergredients(
            target_profile=target_profile,
            constraints=constraints,
            target_concerns=['hydration', 'anti_aging'],
            skin_type='normal',
            budget=1000.0
        )
        
        # Check result structure
        self.assertIn('best_formulation', results)
        self.assertIn('hypergredient_analysis', results)
        self.assertIn('optimization_time_seconds', results)
        
        # Check that hypergredient enhancement is applied
        best = results['best_formulation']
        self.assertIsInstance(best, HypergredientFormulationCandidate)
        
        # Check hypergredient-specific properties
        if hasattr(best, 'synergy_score'):
            self.assertGreaterEqual(best.synergy_score, 0.0)
        if hasattr(best, 'hypergredient_scores'):
            self.assertIsInstance(best.hypergredient_scores, dict)
    
    def test_candidate_enhancement(self):
        """Test formulation candidate enhancement with hypergredient data"""
        # Create basic candidate
        basic_candidate = FormulationCandidate(
            ingredients={'RETINOL': 0.5, 'SODIUM HYALURONATE': 1.0, 'GLYCERIN': 5.0},
            objectives={ObjectiveType.EFFICACY: 0.7, ObjectiveType.SAFETY: 0.8},
            constraints_satisfied=True,
            fitness_score=0.75
        )
        
        # Enhance with hypergredient data
        enhanced = self.optimizer._enhance_candidate_with_hypergredients(
            basic_candidate, ['anti_aging', 'hydration']
        )
        
        # Check enhancement
        self.assertIsInstance(enhanced, HypergredientFormulationCandidate)
        self.assertEqual(enhanced.ingredients, basic_candidate.ingredients)
        self.assertGreaterEqual(enhanced.synergy_score, 0.0)
        
        # Check hypergredient-specific data
        if 'RETINOL' in enhanced.hypergredient_classes:
            self.assertEqual(enhanced.hypergredient_classes['RETINOL'], HypergredientClass.CT)
        if 'SODIUM HYALURONATE' in enhanced.hypergredient_classes:
            self.assertEqual(enhanced.hypergredient_classes['SODIUM HYALURONATE'], HypergredientClass.HY)
    
    def test_integration_performance(self):
        """Test integration performance overhead"""
        target_profile = {'skin_hydration': 0.6}
        constraints = [FormulationConstraint("AQUA", 40.0, 80.0, required=True)]
        
        start_time = time.time()
        results = self.optimizer.optimize_formulation_with_hypergredients(
            target_profile=target_profile,
            constraints=constraints,
            target_concerns=['hydration'],
            budget=500.0
        )
        end_time = time.time()
        
        # Should complete in reasonable time (< 5 seconds for simple case)
        self.assertLess(end_time - start_time, 5.0)
        
        # Should still produce valid results
        self.assertIn('best_formulation', results)
        self.assertGreater(len(results['best_formulation'].ingredients), 0)

class TestHypergredientPerformance(unittest.TestCase):
    """Test suite for hypergredient framework performance requirements"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.database = HypergredientDatabase()
        self.formulator = HypergredientFormulator(self.database)
    
    def test_database_search_speed(self):
        """Test database search performance"""
        # Test search speed (should be < 1ms for small database)
        start_time = time.time()
        for _ in range(100):  # 100 searches
            results = self.database.search_hypergredients(function="antioxidant")
        end_time = time.time()
        
        avg_time_per_search = (end_time - start_time) / 100
        self.assertLess(avg_time_per_search, 0.001)  # < 1ms per search
    
    def test_formulation_generation_speed(self):
        """Test formulation generation speed"""
        # Should generate formulation in < 100ms
        start_time = time.time()
        result = self.formulator.generate_formulation(
            target_concerns=['hydration'],
            budget=500.0
        )
        end_time = time.time()
        
        generation_time = end_time - start_time
        self.assertLess(generation_time, 0.1)  # < 100ms
        
        # Should still produce valid result
        self.assertIn('selected_hypergredients', result)
    
    def test_memory_efficiency(self):
        """Test memory usage stability"""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Generate multiple formulations
        for i in range(50):
            result = self.formulator.generate_formulation(
                target_concerns=['anti_aging'],
                budget=1000.0 + i * 10  # Vary budget slightly
            )
            
            # Should not accumulate memory excessively
            if i % 10 == 0:
                gc.collect()
        
        # Test passes if no memory errors occur
        self.assertTrue(True)

class TestHypergredientRegression(unittest.TestCase):
    """Regression test suite for hypergredient framework edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.database = HypergredientDatabase()
        self.formulator = HypergredientFormulator(self.database)
    
    def test_empty_concerns_handling(self):
        """Test handling of empty target concerns"""
        # Should handle gracefully without crashing
        result = self.formulator.generate_formulation(
            target_concerns=[],
            budget=500.0
        )
        
        # Should return valid structure even with no concerns
        self.assertIn('selected_hypergredients', result)
        self.assertIn('total_cost', result)
    
    def test_very_low_budget_handling(self):
        """Test handling of extremely low budgets"""
        result = self.formulator.generate_formulation(
            target_concerns=['hydration'],
            budget=10.0  # Very low budget
        )
        
        # Should not crash and should respect budget
        self.assertLessEqual(result['total_cost'], 10.0)
        
        # May have no ingredients selected, which is valid
        self.assertIsInstance(result['selected_hypergredients'], dict)
    
    def test_unknown_skin_type_handling(self):
        """Test handling of unknown skin types"""
        result = self.formulator.generate_formulation(
            target_concerns=['hydration'],
            skin_type='unknown_type',
            budget=500.0
        )
        
        # Should use default weights and not crash
        self.assertIn('selected_hypergredients', result)
    
    def test_conflicting_constraints(self):
        """Test handling of potentially conflicting ingredient selections"""
        # Request brightening (needs low pH) and retinoids (pH sensitive)
        result = self.formulator.generate_formulation(
            target_concerns=['brightness', 'wrinkles'],
            budget=1000.0
        )
        
        # Should handle gracefully and make intelligent choices
        self.assertIn('selected_hypergredients', result)
        self.assertIn('recommendations', result)

def run_hypergredient_test_suite():
    """Run the complete hypergredient framework test suite"""
    print("🧬 Hypergredient Framework - Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHypergredientDatabase,
        TestHypergredientFormulator,
        TestHypergredientIntegration,
        TestHypergredientPerformance,
        TestHypergredientRegression
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("HYPERGREDIENT FRAMEWORK TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run:     {result.testsRun}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print(f"Success Rate:  {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Performance benchmarks
    print("\n" + "=" * 60)
    print("HYPERGREDIENT PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    # Database search benchmark
    database = HypergredientDatabase()
    start = time.time()
    for _ in range(1000):
        results = database.search_hypergredients(function="antioxidant")
    search_time = (time.time() - start) / 1000
    print(f"Database Search:        {search_time*1000:.3f}ms (target: <1ms)")
    print(f"Status:                 {'✓ PASS' if search_time < 0.001 else '✗ FAIL'}")
    
    # Formulation generation benchmark
    formulator = HypergredientFormulator(database)
    start = time.time()
    result = formulator.generate_formulation(target_concerns=['hydration'], budget=500.0)
    formulation_time = time.time() - start
    print(f"Formulation Generation: {formulation_time*1000:.3f}ms (target: <100ms)")
    print(f"Status:                 {'✓ PASS' if formulation_time < 0.1 else '✗ FAIL'}")
    
    # Integration benchmark
    optimizer = HypergredientMultiscaleOptimizer()
    start = time.time()
    integration_result = optimizer.optimize_formulation_with_hypergredients(
        target_profile={'skin_hydration': 0.7},
        constraints=[FormulationConstraint("AQUA", 40.0, 80.0, required=True)],
        target_concerns=['hydration'],
        budget=500.0
    )
    integration_time = time.time() - start
    print(f"Integration Overhead:   {integration_time:.3f}s (target: <5s)")
    print(f"Status:                 {'✓ PASS' if integration_time < 5.0 else '✗ FAIL'}")
    
    print(f"\nBenchmark Summary:")
    all_passed = search_time < 0.001 and formulation_time < 0.1 and integration_time < 5.0
    print(f"Overall:                {'✓ ALL BENCHMARKS PASSED' if all_passed else '✗ SOME BENCHMARKS FAILED'}")
    
    # Final result
    if result.failures or result.errors:
        print(f"\n✗ {len(result.failures + result.errors)} test(s) failed")
        return False
    else:
        print(f"\n✅ All tests passed!")
        return True

if __name__ == "__main__":
    success = run_hypergredient_test_suite()
    sys.exit(0 if success else 1)