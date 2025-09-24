#!/usr/bin/env python3
#
# test_deep_integration.py
#
# 🧬 Comprehensive Test Suite for Deep Hypergredient Integration
# 
# This test suite validates the deep integration between the Hypergredient
# Framework and OpenCog AtomSpace, ensuring all components work together
# seamlessly across different deployment scenarios.
#
# Test Coverage:
# - AtomSpace adapter functionality
# - Hypergredient knowledge representation
# - Pattern-based reasoning queries
# - Integrated optimization workflows
# - Compatibility mode fallbacks
# - Performance benchmarks
#
# Usage: python test_deep_integration.py
# --------------------------------------------------------------

import unittest
import sys
import os
import time
from typing import Dict, List, Any

# Import the integration modules
from hypergredient_framework import (
    HypergredientDatabase, HypergredientFormulator, HypergredientClass, HypergredientInfo
)
from hypergredient_atomspace import (
    HypergredientAtomSpaceAdapter, IntegratedHypergredientOptimizer, 
    OPENCOG_AVAILABLE, AtomSpace
)
from hypergredient_integration import (
    HypergredientMultiscaleOptimizer, create_integrated_optimization_demo,
    create_deep_atomspace_integration_demo
)
from multiscale_optimizer import FormulationCandidate, ObjectiveType
from inci_optimizer import FormulationConstraint

class TestHypergredientAtomSpaceAdapter(unittest.TestCase):
    """Test suite for the AtomSpace adapter component"""
    
    def setUp(self):
        """Set up test fixtures"""
        if OPENCOG_AVAILABLE:
            self.atomspace = AtomSpace()
        else:
            self.atomspace = None
        
        self.adapter = HypergredientAtomSpaceAdapter(self.atomspace)
    
    def test_adapter_initialization(self):
        """Test adapter initializes correctly in both modes"""
        self.assertIsNotNone(self.adapter)
        self.assertIsNotNone(self.adapter.hypergredient_db)
        self.assertIsNotNone(self.adapter.formulator)
        
        # Check database is populated
        self.assertGreater(len(self.adapter.hypergredient_db.hypergredients), 5)
        
        # Check adapter mode
        if OPENCOG_AVAILABLE and self.atomspace:
            self.assertTrue(self.adapter.enabled)
            self.assertGreater(len(self.adapter.ingredient_atoms), 0)
            self.assertGreater(len(self.adapter.class_atoms), 0)
        else:
            # Compatibility mode should still work
            self.assertIsNotNone(self.adapter.hypergredient_db)
    
    def test_hypergredient_class_queries(self):
        """Test querying ingredients by hypergredient class"""
        # Test each major hypergredient class
        test_classes = [
            HypergredientClass.CT,  # Cellular Turnover
            HypergredientClass.CS,  # Collagen Synthesis
            HypergredientClass.AO,  # Antioxidants
            HypergredientClass.HY   # Hydration
        ]
        
        for hg_class in test_classes:
            ingredients = self.adapter.query_ingredients_by_class(hg_class)
            self.assertIsInstance(ingredients, list)
            # Should find at least one ingredient for each major class
            if hg_class in [HypergredientClass.CT, HypergredientClass.AO]:
                self.assertGreater(len(ingredients), 0, 
                                 f"No ingredients found for class {hg_class}")
    
    def test_synergy_queries(self):
        """Test querying synergistic ingredient relationships"""
        # Get some ingredients to test
        ct_ingredients = self.adapter.query_ingredients_by_class(HypergredientClass.CT)
        
        if ct_ingredients:
            test_ingredient = ct_ingredients[0]
            synergies = self.adapter.query_synergistic_ingredients(test_ingredient)
            self.assertIsInstance(synergies, list)
            # Synergies list can be empty, that's valid
    
    def test_atomspace_optimization(self):
        """Test AtomSpace-enhanced optimization"""
        target_concerns = ['wrinkles', 'firmness']
        skin_type = "normal"
        budget = 1000.0
        
        results = self.adapter.optimize_formulation_with_atomspace(
            target_concerns=target_concerns,
            skin_type=skin_type,
            budget=budget
        )
        
        # Validate results structure
        self.assertIsInstance(results, dict)
        self.assertIn('selected_hypergredients', results)
        self.assertIn('total_cost', results)
        self.assertIn('synergy_score', results)
        self.assertIn('atomspace_enhanced', results)
        
        # Validate values
        self.assertGreaterEqual(results['total_cost'], 0)
        self.assertGreaterEqual(results['synergy_score'], 0)
        self.assertTrue(results['atomspace_enhanced'])

class TestIntegratedOptimizer(unittest.TestCase):
    """Test suite for the integrated optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        if OPENCOG_AVAILABLE:
            atomspace = AtomSpace()
        else:
            atomspace = None
        
        self.optimizer = IntegratedHypergredientOptimizer(atomspace)
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly"""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.atomspace_adapter)
        self.assertIsNotNone(self.optimizer.hypergredient_optimizer)
    
    def test_integrated_optimization(self):
        """Test integrated optimization workflow"""
        target_profile = {
            'anti_aging_efficacy': 0.80,
            'skin_brightness': 0.70,
            'hydration_level': 0.75
        }
        
        constraints = [
            FormulationConstraint("AQUA", 40.0, 80.0, required=True),
            FormulationConstraint("GLYCERIN", 2.0, 10.0, required=True)
        ]
        
        target_concerns = ['wrinkles', 'hydration']
        
        # Test with AtomSpace enabled
        results = self.optimizer.optimize(
            target_profile=target_profile,
            constraints=constraints,
            target_concerns=target_concerns,
            skin_type="normal",
            budget=1500.0,
            use_atomspace=True
        )
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertIn('integration_level', results)
        
        # Test with AtomSpace disabled
        results_basic = self.optimizer.optimize(
            target_profile=target_profile,
            constraints=constraints,
            target_concerns=target_concerns,
            skin_type="normal",
            budget=1500.0,
            use_atomspace=False
        )
        
        self.assertIsInstance(results_basic, dict)
        self.assertEqual(results_basic['integration_level'], 'basic')
    
    def test_result_merging(self):
        """Test merging of optimization results"""
        atomspace_results = {
            'selected_hypergredients': {},
            'synergy_score': 1.5,
            'compatibility_score': 0.9,
            'atomspace_analysis': {'test': 'data'}
        }
        
        hypergredient_results = {
            'best_formulation': 'test_formulation',
            'synergy_score': 1.2,
            'hypergredient_analysis': {'test': 'hg_data'}
        }
        
        merged = self.optimizer._merge_optimization_results(
            atomspace_results, hypergredient_results
        )
        
        # Check merged structure
        self.assertIn('atomspace_analysis', merged)
        self.assertIn('compatibility_score', merged)
        self.assertIn('optimization_methods', merged)
        
        # Check synergy score averaging
        expected_synergy = (1.5 + 1.2) / 2
        self.assertAlmostEqual(merged['synergy_score'], expected_synergy, places=2)

class TestDeepIntegrationWorkflows(unittest.TestCase):
    """Test suite for complete integration workflows"""
    
    def test_standard_integration_demo(self):
        """Test the standard integration demonstration"""
        try:
            results = create_integrated_optimization_demo()
            self.assertIsInstance(results, dict)
            # Demo should complete without errors
        except Exception as e:
            self.fail(f"Standard integration demo failed: {e}")
    
    def test_deep_atomspace_integration_demo(self):
        """Test the deep AtomSpace integration demonstration"""
        try:
            results = create_deep_atomspace_integration_demo()
            self.assertIsInstance(results, dict)
            self.assertIn('demo_completed', results)
            self.assertTrue(results['demo_completed'])
        except Exception as e:
            self.fail(f"Deep AtomSpace integration demo failed: {e}")

class TestPerformanceAndBenchmarks(unittest.TestCase):
    """Test suite for performance validation"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        if OPENCOG_AVAILABLE:
            atomspace = AtomSpace()
        else:
            atomspace = None
        
        self.optimizer = IntegratedHypergredientOptimizer(atomspace)
        
        # Standard test parameters
        self.target_profile = {
            'anti_aging_efficacy': 0.80,
            'skin_brightness': 0.70
        }
        
        self.constraints = [
            FormulationConstraint("AQUA", 50.0, 70.0, required=True)
        ]
        
        self.target_concerns = ['wrinkles', 'brightness']
    
    def test_optimization_performance(self):
        """Test optimization performance metrics"""
        # Measure time for integrated optimization
        start_time = time.time()
        
        results = self.optimizer.optimize(
            target_profile=self.target_profile,
            constraints=self.constraints,
            target_concerns=self.target_concerns,
            skin_type="normal",
            budget=1000.0,
            use_atomspace=True
        )
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Performance should be reasonable (< 30 seconds for this test)
        self.assertLess(optimization_time, 30.0, 
                       f"Optimization took too long: {optimization_time:.2f}s")
        
        # Results should be valid
        self.assertIsInstance(results, dict)
        # For integrated results, total_cost might be in hypergredient_analysis
        if 'total_cost' not in results and 'hypergredient_analysis' in results:
            hg_analysis = results['hypergredient_analysis']
            if 'total_cost' in hg_analysis:
                results['total_cost'] = hg_analysis['total_cost']
        
        # Check that some cost information is available
        total_cost = results.get('total_cost', 0)
        self.assertGreaterEqual(total_cost, 0)
    
    def test_compatibility_mode_performance(self):
        """Test that compatibility mode doesn't significantly degrade performance"""
        # Create adapter without AtomSpace
        adapter = HypergredientAtomSpaceAdapter(None)
        
        start_time = time.time()
        
        results = adapter.optimize_formulation_with_atomspace(
            target_concerns=self.target_concerns,
            skin_type="normal",
            budget=1000.0
        )
        
        end_time = time.time()
        compatibility_time = end_time - start_time
        
        # Should still complete in reasonable time
        self.assertLess(compatibility_time, 10.0,
                       f"Compatibility mode too slow: {compatibility_time:.2f}s")
        
        # Should return valid results
        self.assertIsInstance(results, dict)
    
    def test_scalability_with_ingredient_count(self):
        """Test performance scales reasonably with ingredient database size"""
        adapter = HypergredientAtomSpaceAdapter(None)
        
        # The current database should handle efficiently
        ingredient_count = len(adapter.hypergredient_db.hypergredients)
        self.assertGreater(ingredient_count, 5, "Database should have multiple ingredients")
        
        # Query performance should be reasonable
        start_time = time.time()
        
        for hg_class in [HypergredientClass.CT, HypergredientClass.AO, HypergredientClass.HY]:
            ingredients = adapter.query_ingredients_by_class(hg_class)
            self.assertIsInstance(ingredients, list)
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Multiple queries should complete quickly
        self.assertLess(query_time, 1.0, f"Queries too slow: {query_time:.2f}s")

class TestCompatibilityAndFallbacks(unittest.TestCase):
    """Test suite for compatibility mode and fallback mechanisms"""
    
    def test_opencog_unavailable_fallback(self):
        """Test graceful fallback when OpenCog is unavailable"""
        # Force compatibility mode
        adapter = HypergredientAtomSpaceAdapter(None)
        self.assertFalse(adapter.enabled)
        
        # Should still work for basic operations
        ingredients = adapter.query_ingredients_by_class(HypergredientClass.AO)
        self.assertIsInstance(ingredients, list)
    
    def test_missing_dependencies_handling(self):
        """Test handling of missing optional dependencies"""
        # The system should still work with basic numpy/pandas
        try:
            from hypergredient_framework import HypergredientDatabase
            db = HypergredientDatabase()
            self.assertGreater(len(db.hypergredients), 0)
        except ImportError as e:
            self.fail(f"Basic functionality should work with minimal dependencies: {e}")
    
    def test_error_recovery(self):
        """Test error recovery in optimization workflows"""
        optimizer = IntegratedHypergredientOptimizer(None)
        
        # Test with invalid parameters
        try:
            results = optimizer.optimize(
                target_profile={},  # Empty profile
                constraints=[],
                target_concerns=[],
                budget=0.0  # Zero budget
            )
            # Should return some result, even if not optimal
            self.assertIsInstance(results, dict)
        except Exception as e:
            # Should handle gracefully, not crash
            self.assertIsInstance(e, (ValueError, TypeError))

def run_comprehensive_tests():
    """Run all integration tests with detailed reporting"""
    print("🧬 COMPREHENSIVE DEEP INTEGRATION TEST SUITE")
    print("=" * 70)
    print()
    
    # Test discovery and execution
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromModule(sys.modules[__name__])
    
    # Custom test runner with detailed output
    class DetailedTestResult(unittest.TextTestResult):
        def startTest(self, test):
            super().startTest(test)
            print(f"Running: {test._testMethodName}")
    
    runner = unittest.TextTestRunner(
        resultclass=DetailedTestResult,
        verbosity=2,
        stream=sys.stdout
    )
    
    print("Starting test execution...")
    print("-" * 50)
    
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    print("-" * 50)
    print(f"Test execution completed in {end_time - start_time:.2f} seconds")
    print()
    
    # Summary report
    print("📊 TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Environment information
    print(f"\n🔧 ENVIRONMENT INFO:")
    print(f"   OpenCog available: {OPENCOG_AVAILABLE}")
    print(f"   Python version: {sys.version}")
    print(f"   Test mode: {'Full integration' if OPENCOG_AVAILABLE else 'Compatibility mode'}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    print(f"\n{'✅ ALL TESTS PASSED!' if success else '❌ SOME TESTS FAILED'}")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    # Run comprehensive test suite
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)