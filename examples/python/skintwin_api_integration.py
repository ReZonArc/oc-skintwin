#!/usr/bin/env python3
#
# skintwin_api_integration.py
#
# 🌐 SkinTwin API Integration - Phase 1 Communication Infrastructure
# 
# This module demonstrates the integration of skintwin formulation data
# with OpenCog communication systems (JSON API, WebSocket, REST endpoints).
# Part of the comprehensive integration roadmap.
#
# Key Features:
# - JSON schema for formulation data exchange
# - RESTful API endpoints for external integrations
# - WebSocket protocol for real-time formulation updates
# - AtomSpace-compatible communication patterns
#
# Part of the OpenCog SkinTwin Integration Roadmap - Phase 1
# --------------------------------------------------------------

import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

# Mock Flask/FastAPI for demonstration (would use actual frameworks in production)
class MockAPIServer:
    """Mock API server for demonstration purposes"""
    def __init__(self):
        self.routes = {}
        self.websocket_handlers = {}
    
    def route(self, path: str, methods: List[str] = ["GET"]):
        def decorator(func):
            self.routes[path] = {"handler": func, "methods": methods}
            return func
        return decorator
    
    def websocket_route(self, path: str):
        def decorator(func):
            self.websocket_handlers[path] = func
            return func
        return decorator

# Import existing skintwin components
from hypergredient_framework import HypergredientClass, HypergredientInfo
from skintwin_storage_integration import FormulationRecord, SkinTwinStorageNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SkinTwinAPIRequest:
    """Standardized API request format for SkinTwin operations"""
    operation: str
    parameters: Dict[str, Any]
    client_id: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class SkinTwinAPIResponse:
    """Standardized API response format for SkinTwin operations"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None
    execution_time_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class SkinTwinJSONSchema:
    """JSON schema definitions for SkinTwin data types"""
    
    @staticmethod
    def formulation_schema() -> Dict[str, Any]:
        """JSON schema for formulation data"""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique formulation identifier"},
                "name": {"type": "string", "description": "Human-readable formulation name"},
                "inci_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of INCI ingredient names"
                },
                "concentrations": {
                    "type": "object",
                    "patternProperties": {
                        "^[A-Z0-9 ]+$": {"type": "number", "minimum": 0, "maximum": 100}
                    },
                    "description": "Ingredient concentrations in percentage"
                },
                "hypergredient_classes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["H.CT", "H.CS", "H.AO", "H.BR", "H.ML", "H.HY", "H.AI", "H.MB", "H.SE", "H.PD"]
                    },
                    "description": "Hypergredient functional classes"
                },
                "performance_scores": {
                    "type": "object",
                    "properties": {
                        "efficacy": {"type": "number", "minimum": 0, "maximum": 1},
                        "safety": {"type": "number", "minimum": 0, "maximum": 1},
                        "stability": {"type": "number", "minimum": 0, "maximum": 1},
                        "cost": {"type": "number", "minimum": 0, "maximum": 1},
                        "regulatory": {"type": "number", "minimum": 0, "maximum": 1},
                        "sustainability": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "description": "Multi-objective performance scores"
                },
                "regulatory_status": {
                    "type": "object",
                    "patternProperties": {
                        "^[A-Z_]+$": {"type": "boolean"}
                    },
                    "description": "Regulatory approval status by region"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional formulation metadata"
                }
            },
            "required": ["id", "name", "inci_list", "concentrations"],
            "additionalProperties": False
        }
    
    @staticmethod
    def api_request_schema() -> Dict[str, Any]:
        """JSON schema for API requests"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "create_formulation",
                        "get_formulation",
                        "update_formulation",
                        "delete_formulation",
                        "search_formulations",
                        "optimize_formulation",
                        "validate_formulation",
                        "analyze_ingredients"
                    ]
                },
                "parameters": {"type": "object"},
                "client_id": {"type": "string"},
                "timestamp": {"type": "number"}
            },
            "required": ["operation", "parameters"],
            "additionalProperties": False
        }
    
    @staticmethod
    def validate_formulation(formulation_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate formulation data against schema"""
        try:
            schema = SkinTwinJSONSchema.formulation_schema()
            # Basic validation (would use jsonschema library in production)
            
            required_fields = schema["required"]
            for field in required_fields:
                if field not in formulation_data:
                    return False, f"Missing required field: {field}"
            
            # Validate INCI list
            if not isinstance(formulation_data["inci_list"], list):
                return False, "inci_list must be an array"
            
            # Validate concentrations sum to reasonable total
            concentrations = formulation_data.get("concentrations", {})
            total_concentration = sum(concentrations.values())
            if total_concentration > 100.0:
                return False, f"Total concentration exceeds 100%: {total_concentration}%"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class SkinTwinAPIServer:
    """OpenCog-compatible API server for SkinTwin operations"""
    
    def __init__(self, storage_node: Optional[SkinTwinStorageNode] = None):
        self.server = MockAPIServer()
        self.storage_node = storage_node or SkinTwinStorageNode()
        self.active_connections = set()
        self.operation_history = []
        
        # Register API routes
        self._register_routes()
        
        logger.info("SkinTwin API Server initialized")
    
    def _register_routes(self):
        """Register all API routes"""
        
        @self.server.route("/api/v1/formulations", ["POST"])
        def create_formulation(request_data: Dict[str, Any]) -> Dict[str, Any]:
            """Create a new formulation"""
            start_time = time.time()
            
            try:
                # Validate request
                is_valid, error_msg = SkinTwinJSONSchema.validate_formulation(request_data)
                if not is_valid:
                    return asdict(SkinTwinAPIResponse(
                        success=False,
                        error=error_msg,
                        execution_time_ms=(time.time() - start_time) * 1000
                    ))
                
                # Create formulation record
                formulation = FormulationRecord(
                    id=request_data["id"],
                    name=request_data["name"],
                    inci_list=request_data["inci_list"],
                    concentrations=request_data["concentrations"],
                    hypergredient_classes=request_data.get("hypergredient_classes", []),
                    performance_scores=request_data.get("performance_scores", {}),
                    regulatory_status=request_data.get("regulatory_status", {}),
                    creation_timestamp=time.time(),
                    last_modified=time.time(),
                    optimization_history=[],
                    metadata=request_data.get("metadata", {})
                )
                
                # Store formulation
                success = self.storage_node.store_formulation(formulation)
                
                if success:
                    self._log_operation("create_formulation", formulation.id, True)
                    return asdict(SkinTwinAPIResponse(
                        success=True,
                        data={"formulation_id": formulation.id, "message": "Formulation created successfully"},
                        execution_time_ms=(time.time() - start_time) * 1000
                    ))
                else:
                    return asdict(SkinTwinAPIResponse(
                        success=False,
                        error="Failed to store formulation",
                        execution_time_ms=(time.time() - start_time) * 1000
                    ))
                
            except Exception as e:
                logger.error(f"Error creating formulation: {e}")
                return asdict(SkinTwinAPIResponse(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000
                ))
        
        @self.server.route("/api/v1/formulations/<formulation_id>", ["GET"])
        def get_formulation(formulation_id: str) -> Dict[str, Any]:
            """Get formulation by ID"""
            start_time = time.time()
            
            try:
                formulation = self.storage_node.load_formulation(formulation_id)
                
                if formulation:
                    self._log_operation("get_formulation", formulation_id, True)
                    return asdict(SkinTwinAPIResponse(
                        success=True,
                        data=asdict(formulation),
                        execution_time_ms=(time.time() - start_time) * 1000
                    ))
                else:
                    return asdict(SkinTwinAPIResponse(
                        success=False,
                        error=f"Formulation not found: {formulation_id}",
                        execution_time_ms=(time.time() - start_time) * 1000
                    ))
                
            except Exception as e:
                logger.error(f"Error retrieving formulation {formulation_id}: {e}")
                return asdict(SkinTwinAPIResponse(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000
                ))
        
        @self.server.route("/api/v1/formulations/search", ["POST"])
        def search_formulations(request_data: Dict[str, Any]) -> Dict[str, Any]:
            """Search formulations by criteria"""
            start_time = time.time()
            
            try:
                search_criteria = request_data.get("criteria", {})
                hypergredient_class = search_criteria.get("hypergredient_class")
                
                if hypergredient_class:
                    formulations = self.storage_node.query_formulations_by_class(hypergredient_class)
                    
                    self._log_operation("search_formulations", hypergredient_class, True)
                    return asdict(SkinTwinAPIResponse(
                        success=True,
                        data={
                            "formulations": [asdict(f) for f in formulations],
                            "count": len(formulations),
                            "search_criteria": search_criteria
                        },
                        execution_time_ms=(time.time() - start_time) * 1000
                    ))
                else:
                    return asdict(SkinTwinAPIResponse(
                        success=False,
                        error="No valid search criteria provided",
                        execution_time_ms=(time.time() - start_time) * 1000
                    ))
                
            except Exception as e:
                logger.error(f"Error searching formulations: {e}")
                return asdict(SkinTwinAPIResponse(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000
                ))
        
        @self.server.route("/api/v1/status", ["GET"])
        def get_status() -> Dict[str, Any]:
            """Get API server status"""
            start_time = time.time()
            
            try:
                stats = self.storage_node.get_performance_statistics()
                
                return asdict(SkinTwinAPIResponse(
                    success=True,
                    data={
                        "server_status": "active",
                        "api_version": "1.0",
                        "storage_stats": stats,
                        "active_connections": len(self.active_connections),
                        "operations_processed": len(self.operation_history),
                        "uptime_seconds": time.time() - start_time
                    },
                    execution_time_ms=(time.time() - start_time) * 1000
                ))
                
            except Exception as e:
                return asdict(SkinTwinAPIResponse(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000
                ))
        
        @self.server.websocket_route("/ws/formulations")
        async def websocket_formulation_updates(websocket):
            """WebSocket endpoint for real-time formulation updates"""
            try:
                self.active_connections.add(websocket)
                logger.info(f"WebSocket connection established. Active connections: {len(self.active_connections)}")
                
                # Send welcome message
                await websocket.send(json.dumps({
                    "type": "connection_established",
                    "message": "Connected to SkinTwin formulation updates",
                    "timestamp": time.time()
                }))
                
                # Keep connection alive and handle messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        response = await self._handle_websocket_message(data)
                        await websocket.send(json.dumps(response))
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format",
                            "timestamp": time.time()
                        }))
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.discard(websocket)
                logger.info(f"WebSocket connection closed. Active connections: {len(self.active_connections)}")
    
    async def _handle_websocket_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket messages"""
        try:
            message_type = data.get("type", "unknown")
            
            if message_type == "subscribe_formulation":
                formulation_id = data.get("formulation_id")
                if formulation_id:
                    # In a real implementation, this would set up subscription
                    return {
                        "type": "subscription_confirmed",
                        "formulation_id": formulation_id,
                        "timestamp": time.time()
                    }
            
            elif message_type == "ping":
                return {
                    "type": "pong",
                    "timestamp": time.time()
                }
            
            else:
                return {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            return {
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            }
    
    def _log_operation(self, operation: str, resource_id: str, success: bool):
        """Log API operations for monitoring"""
        self.operation_history.append({
            "operation": operation,
            "resource_id": resource_id,
            "success": success,
            "timestamp": time.time()
        })
        
        # Keep only last 1000 operations
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]
    
    async def broadcast_formulation_update(self, formulation_id: str, update_type: str, data: Dict[str, Any]):
        """Broadcast formulation updates to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        message = {
            "type": "formulation_update",
            "formulation_id": formulation_id,
            "update_type": update_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # Send to all active connections
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.active_connections -= disconnected

def demonstrate_api_integration():
    """Demonstrate the SkinTwin API integration capabilities"""
    print("🌐 SkinTwin API Integration Demonstration")
    print("=" * 70)
    
    # Initialize API server
    storage_node = SkinTwinStorageNode("/tmp/demo_api_storage.db")
    api_server = SkinTwinAPIServer(storage_node)
    
    print("\n1. JSON Schema Validation...")
    
    # Test JSON schema validation
    valid_formulation = {
        "id": "test_formulation_001",
        "name": "Test Vitamin C Serum",
        "inci_list": ["AQUA", "ASCORBIC ACID", "GLYCERIN"],
        "concentrations": {"AQUA": 80.0, "ASCORBIC ACID": 15.0, "GLYCERIN": 5.0},
        "hypergredient_classes": ["H.AO", "H.HY"],
        "performance_scores": {"efficacy": 0.82, "safety": 0.91},
        "regulatory_status": {"EU": True, "FDA": True}
    }
    
    is_valid, error = SkinTwinJSONSchema.validate_formulation(valid_formulation)
    print(f"  ✓ Valid formulation: {is_valid} - {error if error else 'No errors'}")
    
    # Test invalid formulation
    invalid_formulation = {
        "name": "Invalid Formulation",  # Missing required 'id'
        "inci_list": ["AQUA"],
        "concentrations": {"AQUA": 120.0}  # Exceeds 100%
    }
    
    is_valid, error = SkinTwinJSONSchema.validate_formulation(invalid_formulation)
    print(f"  ✗ Invalid formulation: {is_valid} - {error}")
    
    print("\n2. REST API Operations...")
    
    # Simulate API calls
    create_response = api_server.server.routes["/api/v1/formulations"]["handler"](valid_formulation)
    print(f"  ✓ CREATE: {create_response['success']} - {create_response.get('data', {}).get('message', 'No message')}")
    
    get_response = api_server.server.routes["/api/v1/formulations/<formulation_id>"]["handler"](valid_formulation["id"])
    print(f"  ✓ GET: {get_response['success']} - Retrieved formulation: {get_response.get('data', {}).get('name', 'No name')}")
    
    search_request = {
        "criteria": {
            "hypergredient_class": "H.AO"
        }
    }
    search_response = api_server.server.routes["/api/v1/formulations/search"]["handler"](search_request)
    print(f"  ✓ SEARCH: {search_response['success']} - Found {search_response.get('data', {}).get('count', 0)} formulations")
    
    status_response = api_server.server.routes["/api/v1/status"]["handler"]()
    print(f"  ✓ STATUS: {status_response['success']} - Storage: {status_response.get('data', {}).get('storage_stats', {}).get('total_formulations', 0)} formulations")
    
    print("\n3. Performance Metrics...")
    
    # Performance analysis
    execution_times = [
        create_response.get('execution_time_ms', 0),
        get_response.get('execution_time_ms', 0),
        search_response.get('execution_time_ms', 0),
        status_response.get('execution_time_ms', 0)
    ]
    
    avg_response_time = sum(execution_times) / len(execution_times)
    print(f"  ✓ Average response time: {avg_response_time:.2f} ms")
    print(f"  ✓ Operations processed: {len(api_server.operation_history)}")
    
    print("\n4. API Documentation Sample...")
    
    # Generate API documentation sample
    api_docs = {
        "openapi": "3.0.0",
        "info": {
            "title": "SkinTwin API",
            "version": "1.0.0",
            "description": "OpenCog SkinTwin Formulation API"
        },
        "paths": {
            "/api/v1/formulations": {
                "post": {
                    "summary": "Create a new formulation",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": SkinTwinJSONSchema.formulation_schema()
                            }
                        }
                    }
                }
            }
        }
    }
    
    print(f"  ✓ OpenAPI specification generated")
    print(f"  ✓ Schema validation rules: {len(SkinTwinJSONSchema.formulation_schema()['properties'])} properties")
    
    print("\n📡 API Integration Phase 1 - COMPLETE!")
    print("Next Phase: Implement WebSocket real-time updates and distributed API")

# Mock async demonstration (would use actual async in production)
async def demonstrate_websocket_integration():
    """Demonstrate WebSocket integration (mock implementation)"""
    print("\n5. WebSocket Integration Simulation...")
    
    # Simulate WebSocket connection
    print("  ✓ WebSocket connection established")
    print("  ✓ Subscription to formulation updates active")
    print("  ✓ Real-time update broadcasting ready")
    print("  ✓ Connection management implemented")

if __name__ == "__main__":
    demonstrate_api_integration()
    
    # Run async demonstration
    import asyncio
    asyncio.run(demonstrate_websocket_integration())