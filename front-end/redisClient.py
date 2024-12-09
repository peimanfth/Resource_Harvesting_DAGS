import redis
from redis.exceptions import ConnectionError, TimeoutError
from logger import Logger

class RedisClient:
    def __init__(self, host='127.0.0.1', port=6379, db=0, password=None, pool_size=10, logger=None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=pool_size,
            decode_responses=True
        )
        self.connection = redis.StrictRedis(connection_pool=self.pool)
        self.logger = logger
    def connect(self):
        try:
            self.connection.ping()
            self.logger.info("Connected to Redis server")
        except (ConnectionError, TimeoutError):
            self.logger.info("Failed to connect to Redis server")

    def set_value(self, key, value):
        try:
            self.connection.set(key, value)
            self.logger.info(f"Set {key} to {value}")
        except (ConnectionError, TimeoutError) as e:
            self.logger.info(f"Error setting value: {e}")

    def set_hash(self, key, field, value):
        try:
            self.connection.hset(key, field, value)
            self.logger.info(f"Set {key} field {field} to {value}")
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Error setting hash: {e}")

    def get_value(self, key):
        try:
            value = self.connection.get(key)
            self.logger.info(f"Retrieved {key}: {value}")
            return value
        except (ConnectionError, TimeoutError) as e:
            self.logger.info(f"Error getting value: {e}")
            return None
    
    def get_hash(self, key, field):
        try:
            value = self.connection.hget(key, field)
            self.logger.info(f"Retrieved {key} field {field}: {value}")
            return value
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Error getting hash: {e}")
            return None

    def delete_value(self, key):
        try:
            self.connection.delete(key)
            self.logger.info(f"Deleted {key}")
        except (ConnectionError, TimeoutError) as e:
            self.logger.info(f"Error deleting value: {e}")

    def exists(self, key):
        try:
            exists = self.connection.exists(key)
            self.logger.info(f"Exists {key}: {exists}")
            return exists
        except (ConnectionError, TimeoutError) as e:
            self.logger.info(f"Error checking existence: {e}")
            return False

    def close(self):
        self.pool.disconnect()
        self.logger.info("Connection pool closed")

    def set_dag_utilization(self, dag_name, cpu_usage, memory_usage):
        self.set_hash(dag_name, 'cpu_usage', cpu_usage)
        self.set_hash(dag_name, 'memory_usage', memory_usage)
# Example usage:
if __name__ == "__main__":
    logger = Logger().get_logger('testing')
    client = RedisClient(logger=logger)
    # client.connect()
    # client.set_value("test_key", "test_value")
    # client.get_value("test_key")
    # client.delete_value("test_key")
    # client.exists("test_key")
    # client.set_dag_utilization("AS", 2, 512)
    # client.get_hash("AS", "cpu_usage")
    client.get_value("n_undone_request")
    client.close()
