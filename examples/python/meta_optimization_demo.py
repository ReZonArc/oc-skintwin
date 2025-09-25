#!/usr/bin/env python3
#
# meta_optimization_demo.py
#
# Comprehensive demonstration of the Meta-Optimization Strategy Engine
# showcasing the ability to generate optimal formulations for every possible
# condition and treatment combination.
#
# This script demonstrates:
# - Single condition optimization
# - Batch optimization for all combinations
# - Performance analysis and benchmarking
# - Alternative formulation generation
# - Evidence-based decision making
#
# --------------------------------------------------------------

import time
import json
from collections import defaultdict, Counter
from meta_optimization_engine import (
    MetaOptimizationEngine,
    SkinCondition,
    TreatmentApproach,
    EvidenceLevel
)

def demonstrate_single_condition_optimization():
    """Demonstrate optimization for a single condition with detailed analysis"""
    
    print("🎯 SINGLE CONDITION OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    engine = MetaOptimizationEngine()
    
    # Test case: Advanced anti-aging formulation
    print("\nOptimizing: Advanced Anti-Aging Treatment")
    print("-" * 50)
    
    start_time = time.time()
    result = engine.generate_optimal_formulation(
        condition=SkinCondition.WRINKLES,
        approach=TreatmentApproach.CORRECTION,
        skin_type="normal",
        budget_constraint=1500.0
    )
    optimization_time = time.time() - start_time
    
    # Detailed result analysis
    print(f"\n📊 OPTIMIZATION RESULTS")
    print("-" * 30)
    print(f"Optimization Time: {optimization_time:.3f} seconds")
    print(f"Predicted Efficacy: {result.predicted_efficacy:.1%}")
    print(f"Confidence Score: {result.confidence_score:.2f}/1.0")
    print(f"Strategies Used: {len(result.optimization_strategies_used)}")
    
    # Formulation breakdown
    print(f"\n🧪 FORMULATION BREAKDOWN")
    print("-" * 30)
    sorted_ingredients = sorted(result.formulation.items(), key=lambda x: x[1], reverse=True)
    
    total_actives = 0
    for ingredient, concentration in sorted_ingredients:
        ingredient_type = "💧 Base" if ingredient == "AQUA" else \
                        "🧪 Active" if concentration < 15 and ingredient != "GLYCERIN" else \
                        "🔧 Functional"
        
        if ingredient_type == "🧪 Active":
            total_actives += concentration
            
        print(f"  {ingredient_type} {ingredient:20s}: {concentration:6.2f}%")
    
    print(f"\nTotal Active Concentration: {total_actives:.1f}%")
    
    # Evidence analysis
    print(f"\n📚 EVIDENCE BASIS")
    print("-" * 30)
    for i, evidence in enumerate(result.evidence_basis, 1):
        print(f"  {i}. {evidence}")
    
    # Alternative formulations
    print(f"\n🔄 ALTERNATIVE FORMULATIONS")
    print("-" * 30)
    for alt in result.alternative_formulations:
        print(f"\n{alt['variant_type'].title()} Version:")
        print(f"  Description: {alt['description']}")
        
        # Show top 5 ingredients for alternatives
        alt_sorted = sorted(alt['formulation'].items(), key=lambda x: x[1], reverse=True)[:5]
        for ingredient, conc in alt_sorted:
            print(f"  • {ingredient}: {conc:.1f}%")
    
    return result, optimization_time

def demonstrate_batch_optimization():
    """Demonstrate optimization for all possible condition-treatment combinations"""
    
    print("\n🚀 BATCH OPTIMIZATION FOR ALL CONDITIONS")
    print("=" * 80)
    
    engine = MetaOptimizationEngine()
    
    # Get all available combinations
    all_combinations = engine.knowledge_base.get_all_condition_treatment_pairs()
    
    print(f"Processing {len(all_combinations)} condition-treatment combinations...")
    print("=" * 60)
    
    results = {}
    performance_stats = {
        'total_time': 0,
        'successful_optimizations': 0,
        'failed_optimizations': 0,
        'efficacy_scores': [],
        'confidence_scores': [],
        'formulation_sizes': []
    }
    
    for i, (condition, approach) in enumerate(all_combinations, 1):
        print(f"\n[{i}/{len(all_combinations)}] {condition.value} + {approach.value}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            result = engine.generate_optimal_formulation(condition, approach)
            opt_time = time.time() - start_time
            
            # Store result
            results[(condition, approach)] = result
            
            # Update statistics
            performance_stats['total_time'] += opt_time
            performance_stats['successful_optimizations'] += 1
            performance_stats['efficacy_scores'].append(result.predicted_efficacy)
            performance_stats['confidence_scores'].append(result.confidence_score)
            performance_stats['formulation_sizes'].append(len(result.formulation))
            
            # Summary
            print(f"✅ Success: {result.predicted_efficacy:.1%} efficacy, "
                  f"{result.confidence_score:.2f} confidence, "
                  f"{len(result.formulation)} ingredients ({opt_time:.3f}s)")
            
        except Exception as e:
            performance_stats['failed_optimizations'] += 1
            print(f"❌ Failed: {str(e)}")
    
    # Performance analysis
    print(f"\n📈 BATCH OPTIMIZATION PERFORMANCE")
    print("=" * 50)
    
    success_rate = performance_stats['successful_optimizations'] / len(all_combinations)
    avg_time = performance_stats['total_time'] / len(all_combinations)
    avg_efficacy = sum(performance_stats['efficacy_scores']) / len(performance_stats['efficacy_scores']) if performance_stats['efficacy_scores'] else 0
    avg_confidence = sum(performance_stats['confidence_scores']) / len(performance_stats['confidence_scores']) if performance_stats['confidence_scores'] else 0
    avg_ingredients = sum(performance_stats['formulation_sizes']) / len(performance_stats['formulation_sizes']) if performance_stats['formulation_sizes'] else 0
    
    print(f"Success Rate: {success_rate:.1%} ({performance_stats['successful_optimizations']}/{len(all_combinations)})")
    print(f"Total Time: {performance_stats['total_time']:.2f}s")
    print(f"Average Time per Optimization: {avg_time:.3f}s")
    print(f"Average Predicted Efficacy: {avg_efficacy:.1%}")
    print(f"Average Confidence Score: {avg_confidence:.2f}")
    print(f"Average Formulation Size: {avg_ingredients:.1f} ingredients")
    
    return results, performance_stats

def analyze_formulation_patterns(results):
    """Analyze patterns in the generated formulations"""
    
    print(f"\n🔍 FORMULATION PATTERN ANALYSIS")
    print("=" * 50)
    
    # Ingredient frequency analysis
    ingredient_frequency = Counter()
    condition_ingredients = defaultdict(set)
    approach_ingredients = defaultdict(set)
    
    for (condition, approach), result in results.items():
        for ingredient in result.formulation.keys():
            ingredient_frequency[ingredient] += 1
            condition_ingredients[condition].add(ingredient)
            approach_ingredients[approach].add(ingredient)
    
    # Most common ingredients
    print(f"\n🏆 MOST COMMON INGREDIENTS")
    print("-" * 30)
    for ingredient, count in ingredient_frequency.most_common(10):
        percentage = (count / len(results)) * 100
        print(f"  {ingredient:20s}: {count:2d} formulations ({percentage:4.1f}%)")
    
    # Condition-specific patterns
    print(f"\n🎯 CONDITION-SPECIFIC PATTERNS")
    print("-" * 30)
    for condition in SkinCondition:
        if condition in condition_ingredients:
            specific_ingredients = condition_ingredients[condition]
            common_ingredients = [ing for ing in specific_ingredients 
                                if ingredient_frequency[ing] > len(results) * 0.5]
            unique_ingredients = [ing for ing in specific_ingredients 
                                if ingredient_frequency[ing] <= 2]
            
            print(f"\n{condition.value.title()}:")
            if common_ingredients:
                print(f"  Common: {', '.join(common_ingredients[:5])}")
            if unique_ingredients:
                print(f"  Unique: {', '.join(unique_ingredients[:3])}")
    
    # Efficacy distribution by condition
    print(f"\n📊 EFFICACY BY CONDITION")
    print("-" * 30)
    condition_efficacy = defaultdict(list)
    
    for (condition, approach), result in results.items():
        condition_efficacy[condition].append(result.predicted_efficacy)
    
    for condition, efficacies in condition_efficacy.items():
        avg_efficacy = sum(efficacies) / len(efficacies)
        max_efficacy = max(efficacies)
        min_efficacy = min(efficacies)
        
        print(f"  {condition.value:20s}: "
              f"Avg {avg_efficacy:.1%}, "
              f"Range {min_efficacy:.1%}-{max_efficacy:.1%}")

def benchmark_performance():
    """Benchmark the performance of the meta-optimization engine"""
    
    print(f"\n⚡ PERFORMANCE BENCHMARKING")
    print("=" * 50)
    
    engine = MetaOptimizationEngine()
    
    # Single optimization benchmark
    print("\n🎯 Single Optimization Benchmark")
    print("-" * 35)
    
    times = []
    for i in range(10):
        start_time = time.time()
        result = engine.generate_optimal_formulation(
            SkinCondition.WRINKLES, 
            TreatmentApproach.CORRECTION
        )
        times.append(time.time() - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Average Time: {avg_time:.4f}s")
    print(f"Min Time: {min_time:.4f}s")
    print(f"Max Time: {max_time:.4f}s")
    print(f"Throughput: {1/avg_time:.1f} optimizations/second")
    
    # Memory usage estimation
    print(f"\n💾 Memory Usage Analysis")
    print("-" * 30)
    
    import sys
    engine_size = sys.getsizeof(engine)
    kb_size = sys.getsizeof(engine.knowledge_base)
    history_size = sum(sys.getsizeof(h) for h in engine.learning_history)
    
    total_memory = engine_size + kb_size + history_size
    
    print(f"Engine Size: {engine_size:,} bytes")
    print(f"Knowledge Base: {kb_size:,} bytes")
    print(f"Learning History: {history_size:,} bytes")
    print(f"Total Memory: {total_memory/1024:.1f} KB")
    
    # Scalability test
    print(f"\n📈 Scalability Test")
    print("-" * 25)
    
    batch_sizes = [1, 5, 10]
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        for i in range(batch_size):
            engine.generate_optimal_formulation(
                SkinCondition.ACNE, 
                TreatmentApproach.PREVENTION
            )
        
        batch_time = time.time() - start_time
        time_per_optimization = batch_time / batch_size
        
        print(f"Batch of {batch_size:2d}: {batch_time:.3f}s total, {time_per_optimization:.4f}s per optimization")

def export_results_summary(results, performance_stats):
    """Export comprehensive results summary"""
    
    print(f"\n💾 EXPORTING RESULTS SUMMARY")
    print("=" * 40)
    
    # Create comprehensive summary
    summary = {
        'meta_optimization_summary': {
            'total_combinations_tested': len(results),
            'successful_optimizations': performance_stats['successful_optimizations'],
            'success_rate': performance_stats['successful_optimizations'] / len(results) * 100,
            'total_optimization_time': performance_stats['total_time'],
            'average_time_per_optimization': performance_stats['total_time'] / len(results)
        },
        'performance_metrics': {
            'average_predicted_efficacy': sum(performance_stats['efficacy_scores']) / len(performance_stats['efficacy_scores']) if performance_stats['efficacy_scores'] else 0,
            'average_confidence_score': sum(performance_stats['confidence_scores']) / len(performance_stats['confidence_scores']) if performance_stats['confidence_scores'] else 0,
            'average_formulation_size': sum(performance_stats['formulation_sizes']) / len(performance_stats['formulation_sizes']) if performance_stats['formulation_sizes'] else 0
        },
        'formulation_results': {}
    }
    
    # Add individual results
    for (condition, approach), result in results.items():
        key = f"{condition.value}_{approach.value}"
        summary['formulation_results'][key] = {
            'predicted_efficacy': result.predicted_efficacy,
            'confidence_score': result.confidence_score,
            'ingredient_count': len(result.formulation),
            'top_ingredients': sorted(result.formulation.items(), key=lambda x: x[1], reverse=True)[:5],
            'strategies_used': result.optimization_strategies_used
        }
    
    # Save to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"/tmp/meta_optimization_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Results exported to: {filename}")
    print(f"Summary contains {len(summary['formulation_results'])} formulation results")
    
    return filename, summary

def main():
    """Main demonstration function"""
    
    print("🧠 META-OPTIMIZATION STRATEGY ENGINE")
    print("🔬 Comprehensive Demonstration")
    print("=" * 80)
    print("This demonstration showcases the ability to generate optimal")
    print("formulations for every possible condition and treatment combination.")
    print("=" * 80)
    
    # Part 1: Single condition optimization
    single_result, single_time = demonstrate_single_condition_optimization()
    
    # Part 2: Batch optimization for all combinations
    batch_results, performance_stats = demonstrate_batch_optimization()
    
    # Part 3: Pattern analysis
    analyze_formulation_patterns(batch_results)
    
    # Part 4: Performance benchmarking
    benchmark_performance()
    
    # Part 5: Export results
    export_file, summary = export_results_summary(batch_results, performance_stats)
    
    # Final summary
    print(f"\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 50)
    print(f"✅ Successfully optimized {performance_stats['successful_optimizations']} condition-treatment combinations")
    print(f"✅ Average efficacy prediction: {sum(performance_stats['efficacy_scores'])/len(performance_stats['efficacy_scores']):.1%}")
    print(f"✅ Average optimization time: {performance_stats['total_time']/len(batch_results):.3f}s")
    print(f"✅ Results exported to: {export_file}")
    
    print(f"\n🧠 META-OPTIMIZATION CAPABILITIES DEMONSTRATED:")
    print("  • Generate optimal formulations for ANY condition-treatment combination")
    print("  • Use multiple optimization strategies in parallel")
    print("  • Provide evidence-based recommendations with confidence scoring")
    print("  • Generate alternative formulations for different needs")
    print("  • Learn and adapt from optimization history")
    print("  • Scale efficiently across large numbers of combinations")
    
    return batch_results, performance_stats, summary

if __name__ == "__main__":
    results, stats, summary = main()