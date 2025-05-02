from sqlalchemy import create_engine, Column, Integer, Float, String, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
import os
from typing import Dict, Any, Optional

Base = declarative_base()

class ScenarioRun(Base):
    """Stores information about each scenario analysis run"""
    __tablename__ = 'scenario_runs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    scenario_name = Column(String)
    description = Column(String)
    is_feasible = Column(Boolean)
    
    # Relationships
    parameters = relationship("ScenarioParameter", back_populates="scenario")
    results = relationship("ScenarioResult", back_populates="scenario")
    metrics = relationship("ScenarioMetric", back_populates="scenario")

class ScenarioParameter(Base):
    """Stores the input parameters for each scenario"""
    __tablename__ = 'scenario_parameters'
    
    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey('scenario_runs.id'))
    param_type = Column(String)  # demand_changes, capacity_changes, cost_changes, disruption_params
    param_key = Column(String)   # JSON string of tuple or string key
    param_value = Column(Float)
    
    scenario = relationship("ScenarioRun", back_populates="parameters")

class ScenarioResult(Base):
    """Stores the detailed results of each scenario"""
    __tablename__ = 'scenario_results'
    
    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey('scenario_runs.id'))
    result_type = Column(String)  # production, transportation, inventory, etc.
    entity_from = Column(String)  # facility/store name
    entity_to = Column(String)    # facility/store name
    product = Column(String)
    value = Column(Float)
    
    scenario = relationship("ScenarioRun", back_populates="results")

class ScenarioMetric(Base):
    """Stores calculated metrics for each scenario"""
    __tablename__ = 'scenario_metrics'
    
    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey('scenario_runs.id'))
    metric_type = Column(String)  # cost, utilization, service_level, etc.
    metric_name = Column(String)
    value = Column(Float)
    
    scenario = relationship("ScenarioRun", back_populates="metrics")

def get_db_engine():
    """Create database engine from environment variables"""
    db_user = os.getenv('DB_USER', 'postgres')
    db_pass = os.getenv('DB_PASSWORD', 'postgres')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'supply_chain_analytics')
    
    connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string)

def init_db():
    """Initialize database schema"""
    engine = get_db_engine()
    Base.metadata.create_all(engine)
    return engine

def get_db_session():
    """Create a new database session"""
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def store_scenario_results(scenario_name: str, 
                         description: str,
                         scenario_params: Dict[str, Any],
                         scenario_results: Dict[str, Any]) -> int:
    """
    Store scenario analysis results in the database
    
    Args:
        scenario_name: Name/identifier for the scenario
        description: Description of the scenario
        scenario_params: Dictionary of scenario parameters
        scenario_results: Dictionary of scenario results
        
    Returns:
        scenario_id: ID of the created scenario record
    """
    session = get_db_session()
    
    try:
        # Create scenario run record
        scenario = ScenarioRun(
            scenario_name=scenario_name,
            description=description,
            is_feasible=scenario_results.get('feasible', True)
        )
        session.add(scenario)
        session.flush()  # Get the ID without committing
        
        # Store parameters
        for param_type, params in scenario_params.items():
            for key, value in params.items():
                param = ScenarioParameter(
                    scenario_id=scenario.id,
                    param_type=param_type,
                    param_key=json.dumps(key) if isinstance(key, tuple) else str(key),
                    param_value=float(value)
                )
                session.add(param)
        
        # Store results
        if scenario_results.get('feasible', True):
            # Store production results
            for (facility, product), quantity in scenario_results.get('production', {}).items():
                result = ScenarioResult(
                    scenario_id=scenario.id,
                    result_type='production',
                    entity_from=facility,
                    product=product,
                    value=quantity
                )
                session.add(result)
            
            # Store transportation results
            for (source, dest, product), quantity in scenario_results.get('transportation', {}).items():
                result = ScenarioResult(
                    scenario_id=scenario.id,
                    result_type='transportation',
                    entity_from=source,
                    entity_to=dest,
                    product=product,
                    value=quantity
                )
                session.add(result)
            
            # Store metrics
            for metric_type, metrics in scenario_results.items():
                if metric_type in ['costs', 'capacity_utilization', 'service_level']:
                    for name, value in metrics.items():
                        metric = ScenarioMetric(
                            scenario_id=scenario.id,
                            metric_type=metric_type,
                            metric_name=str(name),
                            value=float(value)
                        )
                        session.add(metric)
        
        session.commit()
        return scenario.id
        
    except Exception as e:
        session.rollback()
        raise e
    
    finally:
        session.close()

def get_scenario_results(scenario_id: int) -> Dict[str, Any]:
    """
    Retrieve scenario results from database
    
    Args:
        scenario_id: ID of the scenario to retrieve
        
    Returns:
        Dictionary containing scenario details and results
    """
    session = get_db_session()
    
    try:
        scenario = session.query(ScenarioRun).get(scenario_id)
        if not scenario:
            raise ValueError(f"No scenario found with ID {scenario_id}")
            
        results = {
            'scenario_name': scenario.scenario_name,
            'description': scenario.description,
            'timestamp': scenario.timestamp.isoformat(),
            'is_feasible': scenario.is_feasible,
            'parameters': {},
            'results': {
                'production': {},
                'transportation': {}
            },
            'metrics': {}
        }
        
        # Get parameters
        for param in scenario.parameters:
            if param.param_type not in results['parameters']:
                results['parameters'][param.param_type] = {}
            key = json.loads(param.param_key) if '{' in param.param_key else param.param_key
            results['parameters'][param.param_type][key] = param.param_value
        
        # Get results
        for result in scenario.results:
            if result.result_type == 'production':
                results['results']['production'][(result.entity_from, result.product)] = result.value
            elif result.result_type == 'transportation':
                results['results']['transportation'][(result.entity_from, result.entity_to, result.product)] = result.value
        
        # Get metrics
        for metric in scenario.metrics:
            if metric.metric_type not in results['metrics']:
                results['metrics'][metric.metric_type] = {}
            results['metrics'][metric.metric_type][metric.metric_name] = metric.value
        
        return results
        
    finally:
        session.close() 