const axios = require('axios')

// local imports
const logger = require(__basedir + '/helpers/logger')

// configurations
const config = require(__basedir + '/config')

class SmartProxy {
  loggerPrefix = '[SPROXY]'
  logLevel = 'debug'
  constructor(workloadConfig) {
    this.setWorkloadConfig(workloadConfig)

    // buffer for incoming requests
    this.requestBuffer = []
    // the next timeout set for the buffer
    this.nextTimeout = -1
  }
  setWorkloadConfig(workloadConfig) {
    this.workloadConfig = workloadConfig
    // we need super fast access to some workload config
    this.upstreamUrl = workloadConfig.upstreamUrl
    this.maxBufferTimeoutMs = workloadConfig.maxBufferTimeoutMs
    this.maxBufferSize = workloadConfig.maxBufferSize
  }
  logReq(message, req) {
    logger.log(this.logLevel, this.loggerPrefix + ' ' + `${message} (request-id ${req.id})`)
  }
  log(message) {
    logger.log(this.logLevel, this.loggerPrefix + ' ' + message)
  }
  proxy(req, res) {
    this.logReq('Received request', req)

    // TODO: extract request body from the req object
    const requestBody = [1, 2, 3, 4]

    // create queue object
    const queueObject = {
      req,
      res,
      requestBody,
    }

    // log queue length
    const queueLength = this.getQueueLength()
    if (queueLength == 0) {
      this.logReq(`queue empty, seeing first request at: ${req.receivedAt}`, req)
    } else {
      this.logReq(`queue length before request: ${queueLength}, time since first request: ${req.receivedAt - this.getFirstRequestReceivedTime()}`, req)
    }

    // record queue length on arrival
    req.queueLengthBefore = queueLength

    // enqueue the request and schedule timeout
    this.requestBuffer.push(queueObject)
    this.scheduleNextTimeout()
  }
  getQueueLength() {
    return this.requestBuffer.length
  }
  getFirstRequestReceivedTime() {
    if (this.getQueueLength() == 0) {
      return null
    } else {
      return this.requestBuffer[0].req.receivedAt
    }
  }
  scheduleNextTimeout() {
    // clear previously scheduled before it happens
    if (this.nextTimeout != -1) {
      clearTimeout(this.nextTimeout)
      this.nextTimeout = -1
    }

    // check if we need to schedule anything
    if (this.requestBuffer.length == 0) {
      this.log('queue empty, nothing to schedule')
      return
    }

    const queueLength = this.getQueueLength()
    // send a request if queue is full, then schedule for next one
    if (queueLength >= this.maxBufferSize) {
      this.log('queue full, *** dispatching ***')
      this.dispatch()
      // function will automatically be called by dispatch
      return
    }

    const elapsedTimeSinceFirst = Date.now() - this.getFirstRequestReceivedTime()
    // TODO: update buffer timeout based on workload
    const nextTimeoutDelay = Math.max(this.maxBufferTimeoutMs - elapsedTimeSinceFirst, 0)
    this.log(`Next scheduled timeout: ${nextTimeoutDelay}`)
    this.nextTimeout = setTimeout(() => {
      this.log('queue timeout, *** dispatching ***')
      this.dispatch()
    }, nextTimeoutDelay)
  }
  // dispatch up to maximum buffer size requests
  dispatch() {
    const queueLength = this.getQueueLength()
    const dispatchLength = Math.min(queueLength, this.maxBufferSize)

    // pop requests from the buffer and update the buffer
    const sendBuffer = this.requestBuffer.slice(0, dispatchLength)
    this.requestBuffer = this.requestBuffer.slice(dispatchLength)

    const sendBufferIds = sendBuffer.map((v) => v.req.id).join(',')
    this.log(`dispatching ids: ${sendBufferIds}`)

    // send the request and respond to the requests
    sendBufferRequest(this.upstreamUrl, sendBuffer, (m) => this.log(m))

    // reschedule next timeout
    this.scheduleNextTimeout()
  }
}

const sendBufferRequest = async (upstreamUrl, sendBuffer, logFunc) => {
  const sendData = sendBuffer.map((v) => v.requestBody)
  try {
    logFunc(`[FETCH] Sending request ${JSON.stringify(sendData)}`)
    const response = await axios.post(upstreamUrl, sendData)
    const data = response.data
    logFunc(`[FETCH] Received response ${JSON.stringify(data)}`)
    
    for (let i=0; i<sendBuffer.length; i++) {
      sendBuffer[i].res.send([data[i]])
    }
  } catch (error) {
    logger.log('error', `[FETCH] error with upstream request: ${error}`)

    sendBuffer.forEach((v) => {
      v.res.status(500).send({
        error
      })
    })
  }

}

module.exports = SmartProxy