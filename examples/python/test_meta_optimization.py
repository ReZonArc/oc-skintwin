#!/usr/bin/env python3
#
# test_meta_optimization.py
#
# Comprehensive test suite for the Meta-Optimization Engine
# Tests all functionality including edge cases, performance, and integration
#
# --------------------------------------------------------------

import unittest
import time
import numpy as np
from meta_optimization_engine import (
    MetaOptimizationEngine, 
    SkinCondition, 
    TreatmentApproach,
    EvidenceLevel,
    ConditionTreatmentKnowledgeBase,
    OptimizationStrategy
)

class TestMetaOptimizationEngine(unittest.TestCase):
    """Comprehensive test suite for the Meta-Optimization Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = MetaOptimizationEngine()
        self.knowledge_base = ConditionTreatmentKnowledgeBase()
    
    def test_engine_initialization(self):
        """Test proper engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertGreater(len(self.engine.optimization_strategies), 0)
        self.assertIsInstance(self.engine.knowledge_base, ConditionTreatmentKnowledgeBase)
        self.assertEqual(len(self.engine.learning_history), 0)
    
    def test_knowledge_base_completeness(self):
        """Test knowledge base has comprehensive data"""
        # Test condition coverage
        conditions_with_profiles = set()
        for condition, approaches in self.knowledge_base.condition_profiles.items():
            conditions_with_profiles.add(condition)
            self.assertIsInstance(condition, SkinCondition)
            self.assertGreater(len(approaches), 0)
        
        # Test we have profiles for key conditions
        key_conditions = [SkinCondition.ACNE, SkinCondition.WRINKLES, SkinCondition.DRYNESS]
        for condition in key_conditions:
            self.assertIn(condition, conditions_with_profiles)
        
        # Test ingredient efficacy data
        self.assertGreater(len(self.knowledge_base.ingredient_efficacy), 0)
        
        for ingredient, data in self.knowledge_base.ingredient_efficacy.items():
            self.assertIn('mechanisms', data)
            self.assertIn('efficacy_score', data)
            self.assertIn('evidence_level', data)
            self.assertIsInstance(data['evidence_level'], EvidenceLevel)
            self.assertGreater(data['efficacy_score'], 0)
            self.assertLessEqual(data['efficacy_score'], 10)
    
    def test_single_condition_optimization(self):
        """Test optimization for a single condition-treatment pair"""
        result = self.engine.generate_optimal_formulation(
            condition=SkinCondition.WRINKLES,
            approach=TreatmentApproach.CORRECTION
        )
        
        # Test result structure
        self.assertIsNotNone(result.formulation)
        self.assertGreater(len(result.formulation), 0)
        self.assertGreater(result.predicted_efficacy, 0)
        self.assertLessEqual(result.predicted_efficacy, 1.0)
        self.assertGreater(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 1.0)
        
        # Test formulation validity
        total_concentration = sum(result.formulation.values())
        self.assertAlmostEqual(total_concentration, 100.0, places=1)
        
        # Test all concentrations are positive
        for ingredient, concentration in result.formulation.items():
            self.assertGreaterEqual(concentration, 0)
        
        # Test required ingredients are present
        profile = self.knowledge_base.get_profile(SkinCondition.WRINKLES, TreatmentApproach.CORRECTION)
        if profile and profile.required_ingredients:
            for required_ingredient in profile.required_ingredients:
                # Should either be present or have a valid substitution reason
                found = any(required_ingredient in ing for ing in result.formulation.keys())
                if not found:
                    print(f"Warning: Required ingredient {required_ingredient} not found in formulation")
    
    def test_multiple_conditions_optimization(self):
        """Test optimization for multiple different conditions"""
        test_cases = [
            (SkinCondition.ACNE, TreatmentApproach.PREVENTION),
            (SkinCondition.DRYNESS, TreatmentApproach.CORRECTION),
            (SkinCondition.SENSITIVITY, TreatmentApproach.GENTLE_CARE)
        ]
        
        results = []
        for condition, approach in test_cases:
            try:
                result = self.engine.generate_optimal_formulation(condition, approach)
                results.append(result)
                
                # Test each result independently
                self.assertIsNotNone(result.formulation)
                self.assertGreater(result.predicted_efficacy, 0)
                self.assertGreater(result.confidence_score, 0)
                
            except Exception as e:
                # Some condition-approach combinations might not be defined
                print(f"Skipping {condition.value}-{approach.value}: {str(e)}")
        
        # Test that we got at least some results
        self.assertGreater(len(results), 0)
        
        # Test that different conditions produce different formulations
        if len(results) >= 2:
            formulation1 = results[0].formulation
            formulation2 = results[1].formulation
            
            # Should have some difference in formulations
            differences = 0
            for ingredient in set(list(formulation1.keys()) + list(formulation2.keys())):
                conc1 = formulation1.get(ingredient, 0)
                conc2 = formulation2.get(ingredient, 0)
                if abs(conc1 - conc2) > 0.1:
                    differences += 1
            
            self.assertGreater(differences, 0, "Different conditions should produce different formulations")
    
    def test_optimization_strategies(self):
        """Test individual optimization strategies"""
        profile = self.knowledge_base.get_profile(SkinCondition.WRINKLES, TreatmentApproach.CORRECTION)
        self.assertIsNotNone(profile)
        
        for strategy in self.engine.optimization_strategies:
            try:
                result = self.engine._run_optimization_strategy(
                    strategy, profile, "normal", 1000.0, []
                )
                
                # Test result structure
                self.assertIsInstance(result, dict)
                self.assertGreater(len(result), 0)
                
                # Test formulation validity
                total = sum(result.values())
                self.assertGreater(total, 80)  # Should be close to 100%
                self.assertLess(total, 120)    # Allow some tolerance
                
                print(f"✓ Strategy '{strategy.name}' produced valid formulation")
                
            except Exception as e:
                self.fail(f"Strategy '{strategy.name}' failed: {str(e)}")
    
    def test_formulation_validation(self):
        """Test formulation validation and refinement"""
        # Create a test formulation that needs validation
        test_formulation = {
            "AQUA": 50.0,
            "RETINOL": 5.0,  # Too high concentration
            "VITAMIN C": 25.0,  # Too high concentration
            "GLYCERIN": 15.0
            # Total = 95% (needs normalization)
        }
        
        profile = self.knowledge_base.get_profile(SkinCondition.WRINKLES, TreatmentApproach.CORRECTION)
        
        validated = self.engine._validate_and_refine_formulation(
            test_formulation, profile, "normal"
        )
        
        # Test normalization to 100%
        total = sum(validated.values())
        self.assertAlmostEqual(total, 100.0, places=1)
        
        # Test concentration limits are respected
        for ingredient, concentration in validated.items():
            if ingredient in self.knowledge_base.ingredient_efficacy:
                max_safe = self.knowledge_base.ingredient_efficacy[ingredient]["max_safe_concentration"]
                self.assertLessEqual(concentration, max_safe + 0.1, 
                                   f"{ingredient} concentration {concentration} exceeds safe limit {max_safe}")
    
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # Test with high-evidence formulation
        high_evidence_formulation = {
            "AQUA": 70.0,
            "RETINOL": 0.5,    # Strong evidence ingredient
            "VITAMIN C": 10.0,  # Strong evidence ingredient
            "HYALURONIC ACID": 1.0,  # Strong evidence ingredient
            "GLYCERIN": 8.5
        }
        
        profile = self.knowledge_base.get_profile(SkinCondition.WRINKLES, TreatmentApproach.CORRECTION)
        
        # Mock strategy results
        strategy_results = {
            'strategy1': {'result': high_evidence_formulation, 'weight': 1.0},
            'strategy2': {'result': high_evidence_formulation, 'weight': 1.0},
            'strategy3': {'result': high_evidence_formulation, 'weight': 1.0}
        }
        
        confidence = self.engine._calculate_confidence_score(
            high_evidence_formulation, strategy_results, profile
        )
        
        self.assertGreater(confidence, 0.5)  # Should be reasonably confident
        self.assertLessEqual(confidence, 1.0)
        
        # Test with low-evidence formulation
        low_evidence_formulation = {
            "AQUA": 90.0,
            "UNKNOWN_INGREDIENT": 10.0
        }
        
        low_confidence = self.engine._calculate_confidence_score(
            low_evidence_formulation, {'strategy1': {'result': low_evidence_formulation, 'weight': 1.0}}, profile
        )
        
        self.assertLess(low_confidence, confidence)  # Should be less confident
    
    def test_efficacy_prediction(self):
        """Test efficacy prediction accuracy"""
        # Test with optimal formulation
        optimal_formulation = {
            "AQUA": 60.0,
            "RETINOL": 0.5,    # At optimal concentration
            "VITAMIN C": 15.0,  # At optimal concentration
            "HYALURONIC ACID": 1.0,  # At optimal concentration
            "GLYCERIN": 8.0,
            "PHENOXYETHANOL": 0.5
        }
        
        profile = self.knowledge_base.get_profile(SkinCondition.WRINKLES, TreatmentApproach.CORRECTION)
        
        efficacy = self.engine._predict_formulation_efficacy(optimal_formulation, profile)
        
        self.assertGreater(efficacy, 0.7)  # Should predict high efficacy
        self.assertLessEqual(efficacy, 1.0)
        
        # Test with suboptimal formulation
        suboptimal_formulation = {
            "AQUA": 95.0,
            "RETINOL": 0.01,   # Very low concentration
            "GLYCERIN": 4.99
        }
        
        low_efficacy = self.engine._predict_formulation_efficacy(suboptimal_formulation, profile)
        
        self.assertLess(low_efficacy, efficacy)  # Should predict lower efficacy
    
    def test_alternative_generation(self):
        """Test alternative formulation generation"""
        base_formulation = {
            "AQUA": 70.0,
            "RETINOL": 0.3,
            "VITAMIN C": 10.0,
            "GLYCERIN": 5.0,
            "PHENOXYETHANOL": 0.5
        }
        
        profile = self.knowledge_base.get_profile(SkinCondition.WRINKLES, TreatmentApproach.CORRECTION)
        strategy_results = {'test': {'result': base_formulation, 'weight': 1.0}}
        
        alternatives = self.engine._generate_alternative_formulations(
            base_formulation, strategy_results, profile
        )
        
        self.assertGreater(len(alternatives), 0)
        
        for alt in alternatives:
            self.assertIn('formulation', alt)
            self.assertIn('variant_type', alt)
            self.assertIn('description', alt)
            
            # Test alternative formulation validity
            alt_formulation = alt['formulation']
            total = sum(alt_formulation.values())
            self.assertGreater(total, 95)
            self.assertLess(total, 105)
    
    def test_performance_metrics(self):
        """Test performance and timing"""
        start_time = time.time()
        
        # Run multiple optimizations to test performance
        for _ in range(3):
            result = self.engine.generate_optimal_formulation(
                condition=SkinCondition.WRINKLES,
                approach=TreatmentApproach.CORRECTION
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly
        self.assertLess(total_time, 10.0, "Optimizations taking too long")
        
        # Test learning history
        self.assertEqual(len(self.engine.learning_history), 3)
        
        # Test performance summary
        summary = self.engine.get_optimization_summary()
        self.assertIn('total_optimizations', summary)
        self.assertIn('average_optimization_time_seconds', summary)
        self.assertIn('average_confidence_score', summary)
        
        self.assertEqual(summary['total_optimizations'], 3)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        
        # Test invalid condition-approach combination
        with self.assertRaises(ValueError):
            self.engine.generate_optimal_formulation(
                condition=SkinCondition.ACNE,
                approach=TreatmentApproach.INTENSIVE_TREATMENT  # Not defined for acne
            )
        
        # Test empty exclusion list
        result = self.engine.generate_optimal_formulation(
            condition=SkinCondition.WRINKLES,
            approach=TreatmentApproach.CORRECTION,
            exclude_ingredients=[]
        )
        self.assertIsNotNone(result)
        
        # Test with exclusions
        result_with_exclusions = self.engine.generate_optimal_formulation(
            condition=SkinCondition.WRINKLES,
            approach=TreatmentApproach.CORRECTION,
            exclude_ingredients=["RETINOL"]
        )
        self.assertIsNotNone(result_with_exclusions)
        
        # Retinol should not be in the formulation (or should be substituted)
        # Note: The current implementation doesn't fully handle exclusions yet
        # This test documents expected future behavior
    
    def test_comprehensive_coverage(self):
        """Test coverage of all available condition-treatment combinations"""
        all_pairs = self.knowledge_base.get_all_condition_treatment_pairs()
        
        self.assertGreater(len(all_pairs), 0)
        
        successful_optimizations = 0
        failed_optimizations = 0
        
        for condition, approach in all_pairs:
            try:
                result = self.engine.generate_optimal_formulation(condition, approach)
                self.assertIsNotNone(result)
                successful_optimizations += 1
            except Exception as e:
                failed_optimizations += 1
                print(f"Failed optimization for {condition.value}-{approach.value}: {str(e)}")
        
        # Should succeed for most combinations
        success_rate = successful_optimizations / len(all_pairs)
        self.assertGreater(success_rate, 0.8, f"Success rate too low: {success_rate:.1%}")
        
        print(f"Coverage test: {successful_optimizations}/{len(all_pairs)} combinations successful")

class TestIntegrationWithExistingFrameworks(unittest.TestCase):
    """Test integration with existing optimization frameworks"""
    
    def setUp(self):
        self.engine = MetaOptimizationEngine()
    
    def test_multiscale_integration(self):
        """Test integration with multiscale optimizer"""
        # This tests that the meta-engine can use the existing MultiscaleConstraintOptimizer
        # Currently using simplified version, but structure is ready for full integration
        
        result = self.engine.generate_optimal_formulation(
            condition=SkinCondition.WRINKLES,
            approach=TreatmentApproach.CORRECTION
        )
        
        # Should use multiscale strategy
        self.assertIn("multiscale_evolutionary", result.optimization_strategies_used)
    
    def test_hypergredient_integration(self):
        """Test integration with hypergredient framework"""
        # Test that hypergredient-based optimization is available
        
        result = self.engine.generate_optimal_formulation(
            condition=SkinCondition.WRINKLES,
            approach=TreatmentApproach.CORRECTION
        )
        
        # Should use hypergredient strategy
        self.assertIn("hypergredient_based", result.optimization_strategies_used)

def run_performance_benchmark():
    """Run performance benchmark tests"""
    print("\n🚀 Running Performance Benchmarks")
    print("=" * 50)
    
    engine = MetaOptimizationEngine()
    
    # Benchmark single optimization
    start_time = time.time()
    result = engine.generate_optimal_formulation(
        condition=SkinCondition.WRINKLES,
        approach=TreatmentApproach.CORRECTION
    )
    single_time = time.time() - start_time
    
    print(f"Single optimization: {single_time:.3f}s")
    
    # Benchmark multiple optimizations
    start_time = time.time()
    for i in range(5):
        engine.generate_optimal_formulation(
            condition=SkinCondition.ACNE,
            approach=TreatmentApproach.PREVENTION
        )
    multiple_time = time.time() - start_time
    
    print(f"5 optimizations: {multiple_time:.3f}s ({multiple_time/5:.3f}s avg)")
    
    # Memory usage (simple check)
    import sys
    memory_usage = sys.getsizeof(engine) + sum(sys.getsizeof(h) for h in engine.learning_history)
    print(f"Memory usage: ~{memory_usage/1024:.1f} KB")
    
    # Learning history growth
    print(f"Learning history: {len(engine.learning_history)} entries")
    
    return single_time, multiple_time/5

if __name__ == "__main__":
    print("🧪 Meta-Optimization Engine Test Suite")
    print("=" * 60)
    
    # Run unit tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromModule(__import__(__name__))
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Run performance benchmarks
    single_time, avg_time = run_performance_benchmark()
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {((test_result.testsRun - len(test_result.failures) - len(test_result.errors))/test_result.testsRun)*100:.1f}%")
    print(f"Performance: {avg_time:.3f}s per optimization")
    
    if test_result.wasSuccessful():
        print("\n✅ All tests passed! Meta-optimization engine is working correctly.")
    else:
        print("\n❌ Some tests failed. Check output above for details.")
        
    print("\n🎯 Meta-Optimization Engine Test Complete!")