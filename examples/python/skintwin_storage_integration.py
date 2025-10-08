#!/usr/bin/env python3
#
# skintwin_storage_integration.py
#
# 🗄️ SkinTwin Storage Integration - Phase 1 Prototype
# 
# This module demonstrates the integration of skintwin formulation data
# with OpenCog storage systems, representing the first implementation
# of the comprehensive integration roadmap.
#
# Key Features:
# - SkinTwinStorageNode for persistent formulation data
# - RocksDB integration for high-performance ingredient databases
# - JSON schema for formulation data exchange
# - Migration utilities for existing formulation databases
#
# Part of the OpenCog SkinTwin Integration Roadmap - Phase 1
# --------------------------------------------------------------

import json
import sqlite3
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Import existing skintwin components
from hypergredient_framework import (
    HypergredientDatabase, HypergredientFormulator, HypergredientClass, 
    HypergredientInfo, RegionType
)
from inci_optimizer import INCIParser, FormulationConstraint
from multiscale_optimizer import FormulationCandidate, ObjectiveType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FormulationRecord:
    """Structured representation of formulation data for storage"""
    id: str
    name: str
    inci_list: List[str]
    concentrations: Dict[str, float]
    hypergredient_classes: List[str]
    performance_scores: Dict[str, float]
    regulatory_status: Dict[str, bool]
    creation_timestamp: float
    last_modified: float
    optimization_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class SkinTwinStorageNode:
    """
    OpenCog-style storage node for SkinTwin formulation data
    
    This class provides persistent storage capabilities for formulation
    data with AtomSpace-compatible interfaces and storage patterns.
    """
    
    def __init__(self, storage_path: str = "/tmp/skintwin_storage.db"):
        """Initialize storage node with SQLite backend (RocksDB placeholder)"""
        self.storage_path = Path(storage_path)
        self.connection = None
        self.hypergredient_db = HypergredientDatabase()
        self.inci_parser = INCIParser()
        
        # Initialize storage schema
        self._initialize_storage()
        
        logger.info(f"SkinTwinStorageNode initialized at {storage_path}")
    
    def _initialize_storage(self):
        """Initialize storage schema and tables"""
        self.connection = sqlite3.connect(str(self.storage_path))
        self.connection.row_factory = sqlite3.Row
        
        # Create formulation storage table
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS formulations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                inci_list TEXT NOT NULL,
                concentrations TEXT NOT NULL,
                hypergredient_classes TEXT NOT NULL,
                performance_scores TEXT NOT NULL,
                regulatory_status TEXT NOT NULL,
                creation_timestamp REAL NOT NULL,
                last_modified REAL NOT NULL,
                optimization_history TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        ''')
        
        # Create ingredients cache table
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS ingredients (
                name TEXT PRIMARY KEY,
                cas_number TEXT,
                inci_name TEXT,
                hypergredient_class TEXT,
                safety_data TEXT,
                regulatory_info TEXT,
                molecular_data TEXT,
                last_updated REAL NOT NULL
            )
        ''')
        
        # Create performance index
        self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_performance 
            ON formulations(performance_scores)
        ''')
        
        self.connection.commit()
        logger.info("Storage schema initialized successfully")
    
    def store_formulation(self, formulation: FormulationRecord) -> bool:
        """Store formulation record with full metadata"""
        try:
            # Serialize complex data structures
            inci_json = json.dumps(formulation.inci_list)
            concentrations_json = json.dumps(formulation.concentrations)
            classes_json = json.dumps(formulation.hypergredient_classes)
            scores_json = json.dumps(formulation.performance_scores)
            regulatory_json = json.dumps(formulation.regulatory_status)
            history_json = json.dumps(formulation.optimization_history)
            metadata_json = json.dumps(formulation.metadata)
            
            # Insert or update formulation
            self.connection.execute('''
                INSERT OR REPLACE INTO formulations 
                (id, name, inci_list, concentrations, hypergredient_classes,
                 performance_scores, regulatory_status, creation_timestamp,
                 last_modified, optimization_history, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                formulation.id, formulation.name, inci_json, concentrations_json,
                classes_json, scores_json, regulatory_json,
                formulation.creation_timestamp, formulation.last_modified,
                history_json, metadata_json
            ))
            
            self.connection.commit()
            logger.info(f"Stored formulation: {formulation.name} (ID: {formulation.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store formulation {formulation.id}: {e}")
            return False
    
    def load_formulation(self, formulation_id: str) -> Optional[FormulationRecord]:
        """Load formulation record by ID"""
        try:
            cursor = self.connection.execute(
                'SELECT * FROM formulations WHERE id = ?',
                (formulation_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Deserialize complex data structures
            return FormulationRecord(
                id=row['id'],
                name=row['name'],
                inci_list=json.loads(row['inci_list']),
                concentrations=json.loads(row['concentrations']),
                hypergredient_classes=json.loads(row['hypergredient_classes']),
                performance_scores=json.loads(row['performance_scores']),
                regulatory_status=json.loads(row['regulatory_status']),
                creation_timestamp=row['creation_timestamp'],
                last_modified=row['last_modified'],
                optimization_history=json.loads(row['optimization_history']),
                metadata=json.loads(row['metadata'])
            )
            
        except Exception as e:
            logger.error(f"Failed to load formulation {formulation_id}: {e}")
            return None
    
    def query_formulations_by_class(self, hypergredient_class: str) -> List[FormulationRecord]:
        """Query formulations containing a specific hypergredient class"""
        try:
            cursor = self.connection.execute(
                'SELECT * FROM formulations WHERE hypergredient_classes LIKE ?',
                (f'%"{hypergredient_class}"%',)
            )
            
            formulations = []
            for row in cursor.fetchall():
                formulation = FormulationRecord(
                    id=row['id'],
                    name=row['name'],
                    inci_list=json.loads(row['inci_list']),
                    concentrations=json.loads(row['concentrations']),
                    hypergredient_classes=json.loads(row['hypergredient_classes']),
                    performance_scores=json.loads(row['performance_scores']),
                    regulatory_status=json.loads(row['regulatory_status']),
                    creation_timestamp=row['creation_timestamp'],
                    last_modified=row['last_modified'],
                    optimization_history=json.loads(row['optimization_history']),
                    metadata=json.loads(row['metadata'])
                )
                formulations.append(formulation)
            
            logger.info(f"Found {len(formulations)} formulations with class {hypergredient_class}")
            return formulations
            
        except Exception as e:
            logger.error(f"Failed to query formulations by class {hypergredient_class}: {e}")
            return []
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get storage performance statistics"""
        try:
            # Count total formulations
            cursor = self.connection.execute('SELECT COUNT(*) as count FROM formulations')
            total_formulations = cursor.fetchone()['count']
            
            # Get storage size
            storage_size = self.storage_path.stat().st_size if self.storage_path.exists() else 0
            
            # Calculate average access time
            start_time = time.time()
            cursor = self.connection.execute('SELECT id FROM formulations LIMIT 10')
            cursor.fetchall()
            avg_access_time = (time.time() - start_time) / 10
            
            return {
                'total_formulations': total_formulations,
                'storage_size_bytes': storage_size,
                'storage_size_mb': storage_size / (1024 * 1024),
                'average_access_time_ms': avg_access_time * 1000,
                'storage_path': str(self.storage_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance statistics: {e}")
            return {}

class SkinTwinMigrationUtility:
    """Utility for migrating existing formulation data to new storage format"""
    
    def __init__(self, storage_node: SkinTwinStorageNode):
        self.storage_node = storage_node
        self.hypergredient_db = HypergredientDatabase()
        self.inci_parser = INCIParser()
    
    def migrate_from_legacy_format(self, legacy_data: List[Dict[str, Any]]) -> int:
        """Migrate formulations from legacy format to storage node"""
        migrated_count = 0
        
        for legacy_formulation in legacy_data:
            try:
                # Convert to new format
                formulation_id = legacy_formulation.get('id', f"legacy_{int(time.time())}_{migrated_count}")
                
                # Parse INCI list
                inci_list = legacy_formulation.get('ingredients', [])
                
                # Analyze with hypergredient framework
                hypergredient_classes = []
                concentrations = {}
                
                for ingredient in inci_list:
                    ingredient_name = ingredient if isinstance(ingredient, str) else ingredient.get('name', '')
                    concentration = 1.0 if isinstance(ingredient, str) else ingredient.get('concentration', 1.0)
                    
                    concentrations[ingredient_name] = concentration
                    
                    # Classify ingredient (simplified for demo)
                    if 'ACID' in ingredient_name.upper():
                        if 'H.AO' not in hypergredient_classes:
                            hypergredient_classes.append('H.AO')
                    elif 'GLYCERIN' in ingredient_name.upper():
                        if 'H.HY' not in hypergredient_classes:
                            hypergredient_classes.append('H.HY')
                
                # Create formulation record
                formulation = FormulationRecord(
                    id=formulation_id,
                    name=legacy_formulation.get('name', f'Legacy Formulation {migrated_count}'),
                    inci_list=inci_list,
                    concentrations=concentrations,
                    hypergredient_classes=hypergredient_classes,
                    performance_scores=legacy_formulation.get('scores', {}),
                    regulatory_status=legacy_formulation.get('regulatory', {}),
                    creation_timestamp=legacy_formulation.get('created', time.time()),
                    last_modified=time.time(),
                    optimization_history=[],
                    metadata=legacy_formulation.get('metadata', {})
                )
                
                # Store in new format
                if self.storage_node.store_formulation(formulation):
                    migrated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate formulation {migrated_count}: {e}")
        
        logger.info(f"Successfully migrated {migrated_count} formulations")
        return migrated_count

def demonstrate_storage_integration():
    """Demonstrate the SkinTwin storage integration capabilities"""
    print("🗄️ SkinTwin Storage Integration Demonstration")
    print("=" * 70)
    
    # Initialize storage node
    storage_node = SkinTwinStorageNode("/tmp/demo_skintwin_storage.db")
    
    print("\n1. Creating sample formulations...")
    
    # Create sample formulations
    sample_formulations = [
        FormulationRecord(
            id="anti_aging_serum_001",
            name="Advanced Anti-Aging Serum",
            inci_list=["AQUA", "GLYCERIN", "NIACINAMIDE", "RETINOL", "HYALURONIC ACID"],
            concentrations={"AQUA": 65.0, "GLYCERIN": 15.0, "NIACINAMIDE": 10.0, "RETINOL": 5.0, "HYALURONIC ACID": 5.0},
            hypergredient_classes=["H.CT", "H.HY", "H.CS"],
            performance_scores={"efficacy": 0.85, "safety": 0.92, "stability": 0.78},
            regulatory_status={"EU": True, "FDA": True, "HEALTH_CANADA": True},
            creation_timestamp=time.time(),
            last_modified=time.time(),
            optimization_history=[{"iteration": 1, "score": 0.85}],
            metadata={"formulator": "SkinTwin AI", "target_age": "35-55", "skin_type": "normal"}
        ),
        FormulationRecord(
            id="hydrating_moisturizer_001",
            name="Deep Hydration Moisturizer",
            inci_list=["AQUA", "GLYCERIN", "CERAMIDE NP", "SODIUM HYALURONATE", "SQUALANE"],
            concentrations={"AQUA": 70.0, "GLYCERIN": 12.0, "CERAMIDE NP": 8.0, "SODIUM HYALURONATE": 5.0, "SQUALANE": 5.0},
            hypergredient_classes=["H.HY", "H.BR"],
            performance_scores={"efficacy": 0.88, "safety": 0.95, "stability": 0.82},
            regulatory_status={"EU": True, "FDA": True, "HEALTH_CANADA": True},
            creation_timestamp=time.time(),
            last_modified=time.time(),
            optimization_history=[{"iteration": 1, "score": 0.88}],
            metadata={"formulator": "SkinTwin AI", "target_age": "all", "skin_type": "dry"}
        )
    ]
    
    # Store formulations
    for formulation in sample_formulations:
        success = storage_node.store_formulation(formulation)
        print(f"  ✓ Stored: {formulation.name} - {'Success' if success else 'Failed'}")
    
    print("\n2. Loading formulations...")
    
    # Load and display formulations
    for formulation in sample_formulations:
        loaded = storage_node.load_formulation(formulation.id)
        if loaded:
            print(f"  ✓ Loaded: {loaded.name}")
            print(f"    - Classes: {', '.join(loaded.hypergredient_classes)}")
            print(f"    - Performance: {loaded.performance_scores}")
        else:
            print(f"  ✗ Failed to load: {formulation.id}")
    
    print("\n3. Querying by hypergredient class...")
    
    # Query by hypergredient class
    hydration_formulations = storage_node.query_formulations_by_class("H.HY")
    print(f"  Found {len(hydration_formulations)} formulations with hydration properties:")
    for formulation in hydration_formulations:
        print(f"    - {formulation.name}")
    
    print("\n4. Performance statistics...")
    
    # Get performance statistics
    stats = storage_node.get_performance_statistics()
    print(f"  ✓ Total formulations: {stats.get('total_formulations', 0)}")
    print(f"  ✓ Storage size: {stats.get('storage_size_mb', 0):.2f} MB")
    print(f"  ✓ Average access time: {stats.get('average_access_time_ms', 0):.2f} ms")
    
    print("\n5. Migration demonstration...")
    
    # Demonstrate migration utility
    migration_utility = SkinTwinMigrationUtility(storage_node)
    
    # Sample legacy data
    legacy_data = [
        {
            "id": "legacy_001",
            "name": "Legacy Vitamin C Serum",
            "ingredients": ["AQUA", "ASCORBIC ACID", "GLYCERIN", "TOCOPHEROL"],
            "scores": {"efficacy": 0.75, "safety": 0.88},
            "regulatory": {"EU": True, "FDA": False},
            "created": time.time() - 86400  # 1 day ago
        }
    ]
    
    migrated = migration_utility.migrate_from_legacy_format(legacy_data)
    print(f"  ✓ Migrated {migrated} legacy formulations")
    
    # Final statistics
    final_stats = storage_node.get_performance_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"  Total formulations: {final_stats.get('total_formulations', 0)}")
    print(f"  Storage efficiency: {final_stats.get('average_access_time_ms', 0):.2f} ms avg access")
    
    print("\n🎯 Storage Integration Phase 1 - COMPLETE!")
    print("Next Phase: Implement RocksDB backend and distributed storage")

if __name__ == "__main__":
    demonstrate_storage_integration()