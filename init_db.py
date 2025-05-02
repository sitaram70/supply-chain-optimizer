from db_schema import init_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database"""
    try:
        logger.info("Initializing database...")
        engine = init_db()
        logger.info("Database initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    main() 