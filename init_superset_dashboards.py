import json
import os
from superset import app, db
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.models.core import Database
from superset.connectors.sqla.models import SqlaTable, TableColumn

def create_supply_chain_database():
    """Create database connection in Superset"""
    database = Database(
        database_name='Supply Chain Analytics',
        sqlalchemy_uri=os.getenv('SQLALCHEMY_DATABASE_URI', 
                               'postgresql://postgres:postgres@db:5432/supply_chain_analytics')
    )
    db.session.add(database)
    db.session.commit()
    return database

def create_scenario_tables(database):
    """Create table objects for scenario analysis tables"""
    tables = {}
    
    # Scenario Runs table
    scenario_runs = SqlaTable(
        table_name='scenario_runs',
        database=database,
        schema=None,
        columns=[
            TableColumn(column_name='id', type='INTEGER'),
            TableColumn(column_name='timestamp', type='TIMESTAMP'),
            TableColumn(column_name='scenario_name', type='STRING'),
            TableColumn(column_name='description', type='STRING'),
            TableColumn(column_name='is_feasible', type='BOOLEAN')
        ]
    )
    tables['scenario_runs'] = scenario_runs
    
    # Scenario Parameters table
    scenario_params = SqlaTable(
        table_name='scenario_parameters',
        database=database,
        schema=None,
        columns=[
            TableColumn(column_name='id', type='INTEGER'),
            TableColumn(column_name='scenario_id', type='INTEGER'),
            TableColumn(column_name='param_type', type='STRING'),
            TableColumn(column_name='param_key', type='STRING'),
            TableColumn(column_name='param_value', type='FLOAT')
        ]
    )
    tables['scenario_parameters'] = scenario_params
    
    # Scenario Results table
    scenario_results = SqlaTable(
        table_name='scenario_results',
        database=database,
        schema=None,
        columns=[
            TableColumn(column_name='id', type='INTEGER'),
            TableColumn(column_name='scenario_id', type='INTEGER'),
            TableColumn(column_name='result_type', type='STRING'),
            TableColumn(column_name='entity_from', type='STRING'),
            TableColumn(column_name='entity_to', type='STRING'),
            TableColumn(column_name='product', type='STRING'),
            TableColumn(column_name='value', type='FLOAT')
        ]
    )
    tables['scenario_results'] = scenario_results
    
    # Scenario Metrics table
    scenario_metrics = SqlaTable(
        table_name='scenario_metrics',
        database=database,
        schema=None,
        columns=[
            TableColumn(column_name='id', type='INTEGER'),
            TableColumn(column_name='scenario_id', type='INTEGER'),
            TableColumn(column_name='metric_type', type='STRING'),
            TableColumn(column_name='metric_name', type='STRING'),
            TableColumn(column_name='value', type='FLOAT')
        ]
    )
    tables['scenario_metrics'] = scenario_metrics
    
    # Add tables to session and commit
    for table in tables.values():
        db.session.add(table)
    db.session.commit()
    
    return tables

def create_default_charts(tables):
    """Create default charts for supply chain analytics"""
    charts = []
    
    # Scenario Comparison Chart
    scenario_comparison = Slice(
        slice_name="Scenario Cost Comparison",
        viz_type='bar',
        datasource_type='table',
        datasource_id=tables['scenario_metrics'].id,
        params=json.dumps({
            "viz_type": "bar",
            "groupby": ["scenario_id"],
            "metrics": ["value"],
            "adhoc_filters": [
                {
                    "clause": "WHERE",
                    "expressionType": "SIMPLE",
                    "filterOptionName": "filter_8",
                    "comparator": "costs",
                    "operator": "==",
                    "subject": "metric_type"
                }
            ],
            "row_limit": 10000,
            "time_range": "No filter",
            "order_desc": True,
            "show_legend": True,
            "show_bar_value": True,
            "color_scheme": "supersetColors"
        })
    )
    charts.append(scenario_comparison)
    
    # Service Level Chart
    service_level = Slice(
        slice_name="Service Level by Store",
        viz_type='line',
        datasource_type='table',
        datasource_id=tables['scenario_metrics'].id,
        params=json.dumps({
            "viz_type": "line",
            "metrics": ["value"],
            "groupby": ["metric_name"],
            "adhoc_filters": [
                {
                    "clause": "WHERE",
                    "expressionType": "SIMPLE",
                    "filterOptionName": "filter_7",
                    "comparator": "service_level",
                    "operator": "==",
                    "subject": "metric_type"
                }
            ],
            "row_limit": 10000,
            "time_range": "No filter",
            "show_legend": True
        })
    )
    charts.append(service_level)
    
    # Capacity Utilization Chart
    capacity_util = Slice(
        slice_name="Capacity Utilization by Facility",
        viz_type='heatmap',
        datasource_type='table',
        datasource_id=tables['scenario_metrics'].id,
        params=json.dumps({
            "viz_type": "heatmap",
            "all_columns_x": ["metric_name"],
            "all_columns_y": ["scenario_id"],
            "metric": "value",
            "adhoc_filters": [
                {
                    "clause": "WHERE",
                    "expressionType": "SIMPLE",
                    "filterOptionName": "filter_9",
                    "comparator": "capacity_utilization",
                    "operator": "==",
                    "subject": "metric_type"
                }
            ],
            "row_limit": 10000,
            "time_range": "No filter"
        })
    )
    charts.append(capacity_util)
    
    # Add charts to session and commit
    for chart in charts:
        db.session.add(chart)
    db.session.commit()
    
    return charts

def create_default_dashboard(charts):
    """Create default supply chain analytics dashboard"""
    dashboard = Dashboard(
        dashboard_title="Supply Chain Analytics",
        position_json=json.dumps({
            "CHART-explorer-1": {
                "type": "CHART",
                "id": "CHART-explorer-1",
                "children": [],
                "meta": {
                    "width": 12,
                    "height": 50,
                    "chartId": charts[0].id
                }
            },
            "CHART-explorer-2": {
                "type": "CHART",
                "id": "CHART-explorer-2",
                "children": [],
                "meta": {
                    "width": 6,
                    "height": 50,
                    "chartId": charts[1].id
                }
            },
            "CHART-explorer-3": {
                "type": "CHART",
                "id": "CHART-explorer-3",
                "children": [],
                "meta": {
                    "width": 6,
                    "height": 50,
                    "chartId": charts[2].id
                }
            }
        }),
        slices=charts
    )
    db.session.add(dashboard)
    db.session.commit()
    return dashboard

def main():
    """Initialize Superset with supply chain analytics dashboards"""
    with app.app_context():
        # Create database connection
        database = create_supply_chain_database()
        
        # Create table objects
        tables = create_scenario_tables(database)
        
        # Create charts
        charts = create_default_charts(tables)
        
        # Create dashboard
        dashboard = create_default_dashboard(charts)
        
        print("Successfully initialized Superset with supply chain analytics dashboards!")

if __name__ == "__main__":
    main() 