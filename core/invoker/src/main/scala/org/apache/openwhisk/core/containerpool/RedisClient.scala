package org.apache.openwhisk.core.containerpool


import redis.clients.jedis.{JedisPool, Jedis}
import org.apache.commons.pool2.impl.GenericObjectPoolConfig
import org.apache.openwhisk.common.Logging
import akka.compat.Future
import scala.concurrent.Future
import scala.collection.JavaConverters._

class RedisClient (
    host: String = "172.17.0.1",
    port: Int = 6379,
    password: String = "openwhisk",
    database: Int = 0,
    logging: Logging,
){
    val poolConfig = new GenericObjectPoolConfig[Jedis]()
    poolConfig.setMaxTotal(300)
    poolConfig.setMaxIdle(100)
    poolConfig.setMinIdle(1)
    // val poolConfig1 = new GenericObjectPoolConfig[Connection]()
    // poolConfig.setMaxTotal(300)
    // poolConfig.setMaxIdle(100)
    // poolConfig.setMinIdle(1)
    val pool = new JedisPool(poolConfig, host, port, 2000, null, database)
    // val jedis = new JedisPooled(poolConfig1, host, port)
    // val pool =  new JedisPool(poolConfig, host, port)
    logging.info(this, s"RedisClient connects to: host: $host, port: $port, password: $password, database: $database")

    def setActivations(key: String, value: String): Unit = {
        val jedis = pool.getResource
        jedis.set(key, value)
        jedis.close()
    }
    
    def setActivationsByController(key: String = "activeActivationsByController", value: Future[List[(String, String)]]): Unit = {
        value.andThen({
            case scala.util.Success(list) => {
                val jedis = pool.getResource
                //set the length of the list and all the values to redis
                // jedis.set("number of inflight activations", list.length.toString())
                list.foreach({case (k, v) => jedis.hset(key, k, v)})
                jedis.close()
            }
            case scala.util.Failure(e) => logging.error(this, s"setActivationsByController failed: $e")
        })(scala.concurrent.ExecutionContext.global)
    }

    def setActivationsByInvoker(key: String = "activeActivationsByInvoker",invokerID: String, value: Future[Int]): Unit = {
        value.andThen({
            case scala.util.Success(_: Int) => {
                if (value.value.get.get != 0) {
                   
                    val jedis = pool.getResource
                    jedis.hset(key, invokerID, value.toString())
                    jedis.close()
                }
            }
            case scala.util.Failure(e) => logging.error(this, s"setActivationsByInvoker failed: $e")
        })(scala.concurrent.ExecutionContext.global)
    }

    def setActivationsByAction(key: String = "activeActivationsByAction", actionName: String, value: Future[Int]): Unit = {
        value.andThen({
            case scala.util.Success(_: Int) => {
                if (value.value.get.get != 0) {
                    val jedis = pool.getResource
                    jedis.hset(key, actionName, value.toString())
                    jedis.close()
                }
            }
            case scala.util.Failure(e) => logging.error(this, s"setActivationsByAction failed: $e")
        })(scala.concurrent.ExecutionContext.global)
    }
    
    def setTestValue(key: String = "testKey", value: String = "testValue"): Unit = {
        val jedis = pool.getResource
        jedis.set(key, value)
        jedis.close()
    }

    def logResourceUtil(functionName: String, cpuUtil: String, memoryUtil: String, timestamp: String): Unit = {
        val jedis = pool.getResource
        try {
            val key = s"resourceUtil:$functionName" // Prefix the function name to create a unique key
            val hashFields = Map(
                "timestamp" -> timestamp,
                "cpuUtil" -> cpuUtil,
                "memoryUtil" -> memoryUtil
            ).asJava
            // Use hmset to set multiple hash fields at once
            jedis.hmset(key, hashFields)
        } catch {
            case e: Exception => logging.error(this, s"logResourceUtil failed: $e")
        } finally {
            jedis.close() // Ensure jedis is closed after operation
        }
    }


    

    //     val jedis = pool.getResource
    //    //set to redis using key and deconstruct the future list as values
    //     jedis.set(key, value.toString())
    //     jedis.close()
    

    // def setActivationsInvoker(key: String, value: String): Unit = {
    //     val jedis = pool.getResource
    //     jedis.hset()
    // }

}