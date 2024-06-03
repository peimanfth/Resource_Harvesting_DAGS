import redis
from redis.exceptions import ConnectionError, TimeoutError

class RedisClient:
    def __init__(self, host='127.0.0.1', port=6379, db=0, password=None, pool_size=10):
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

    def connect(self):
        try:
            self.connection.ping()
            print("Connected to Redis server")
        except (ConnectionError, TimeoutError):
            print("Failed to connect to Redis server")

    def set_value(self, key, value):
        try:
            self.connection.set(key, value)
            print(f"Set {key} to {value}")
        except (ConnectionError, TimeoutError) as e:
            print(f"Error setting value: {e}")

    def get_value(self, key):
        try:
            value = self.connection.get(key)
            print(f"Retrieved {key}: {value}")
            return value
        except (ConnectionError, TimeoutError) as e:
            print(f"Error getting value: {e}")
            return None

    def delete_value(self, key):
        try:
            self.connection.delete(key)
            print(f"Deleted {key}")
        except (ConnectionError, TimeoutError) as e:
            print(f"Error deleting value: {e}")

    def exists(self, key):
        try:
            exists = self.connection.exists(key)
            print(f"Exists {key}: {exists}")
            return exists
        except (ConnectionError, TimeoutError) as e:
            print(f"Error checking existence: {e}")
            return False

    def close(self):
        self.pool.disconnect()
        print("Connection pool closed")

# Example usage:
if __name__ == "__main__":
    client = RedisClient(password="openwhisk")
    client.connect()
    client.set_value("test_key", "test_value")
    client.get_value("test_key")
    client.delete_value("test_key")
    client.exists("test_key")
    client.close()
