package org.apache.openwhisk.core.loadBalancer

object Params {
  val redisHost: String = "192.168.1.103"
  val redisPort: Int = 6379
  val redisPassword: Option[String] = None
  val redisDatabase: Int = 0
}
