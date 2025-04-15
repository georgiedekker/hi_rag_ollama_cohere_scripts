from neo4j import GraphDatabase
import json
import sys
import os

# Usage: python import_to_neo4j.py /path/to/combined_entities.json
# Neo4j connection details (update or use environment variables)
NEO4J_URI = os.getenv("NEO4J_URL", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def import_to_neo4j(json_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Clear existing data (optional)
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create entities as nodes
        for entity in data.get('entities', []):
            # Build properties dict from entity attributes
            props = {k: v for k, v in entity.items() if k not in ['id', 'type']}
            
            # Create node with appropriate label and properties
            query = (
                f"CREATE (e:{entity.get('type', 'Entity')} {{id: $id, name: $name, "
                + ", ".join([f"{k}: ${k}" for k in props.keys()])
                + "}})"
            )
            
            params = {"id": entity.get('id'), "name": entity.get('name'), **props}
            session.run(query, params)
        
        # Create relationships
        for rel in data.get('relationships', []):
            query = (
                "MATCH (s {id: $source}), (t {id: $target}) "
                "CREATE (s)-[r:" + rel.get('type', 'RELATED_TO') + "]->(t)"
            )
            
            params = {
                "source": rel.get('source'),
                "target": rel.get('target')
            }
            session.run(query, params)
    
    driver.close()
    print(f"Imported {len(data.get('entities', []))} entities and {len(data.get('relationships', []))} relationships")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_to_neo4j.py /path/to/combined_entities.json")
        sys.exit(1) # Exit if no argument provided
    
    import_to_neo4j(sys.argv[1])