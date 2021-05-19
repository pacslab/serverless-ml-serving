const axios = require('axios')

// local imports
const logger = require(__basedir + '/helpers/logger')

// configurations
const config = require(__basedir + '/config')

const headerPrefix = 'X-SmartProxy-'

class SmartProxy {
  loggerPrefix = '[SPROXY]'
  logLevel = 'debug'
  constructor(workloadConfig, smartMonitor) {
    // smart monitor object
    this.smartMonitor = smartMonitor

    // setting workload config
    this.setWorkloadConfig(workloadConfig)

    // buffer for incoming requests
    this.requestBuffer = []
    // the next timeout set for the buffer
    this.nextTimeout = -1
  }
  setWorkloadConfig(workloadConfig) {
    this.workloadConfig = workloadConfig
    this.smartMonitor.setWorkloadConfig(workloadConfig)
    // we need super fast access to some workload config
    this.upstreamUrl = workloadConfig.upstreamUrl
    this.maxBufferTimeoutMs = workloadConfig.maxBufferTimeoutMs
    this.maxBufferSize = workloadConfig.maxBufferSize
    this.isTFServing = workloadConfig.isTFServing
  }
  logReq(message, req) {
    logger.log(this.logLevel, this.loggerPrefix + ' ' + `${message} (request-id ${req.id})`)
  }
  log(message) {
    logger.log(this.logLevel, this.loggerPrefix + ' ' + message)
  }
  proxy(req, res) {
    this.logReq('Received request', req)
    this.smartMonitor.recordArrival()

    // extract request body from the req object
    let requestBody = Array.isArray(req.body[0]) ? req.body[0] : req.body
    if (this.isTFServing) {
      requestBody = requestBody.instances[0]
    } else {
      if (Array.isArray(requestBody)) {
        requestBody = ((typeof(requestBody[0]) === 'number') ? requestBody : requestBody[0])
      }
    }

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
    req.respHeader = {}
    req.respHeader[headerPrefix + 'queuePosition'] = queueLength
    req.respHeader[headerPrefix + 'receivedAt'] = req.receivedAt

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
      this.dispatch(false)
      // function will automatically be called by dispatch
      return
    }

    const elapsedTimeSinceFirst = Date.now() - this.getFirstRequestReceivedTime()
    // TODO: update buffer timeout based on workload
    const nextTimeoutDelay = Math.max(this.maxBufferTimeoutMs - elapsedTimeSinceFirst, 0)
    this.log(`Next scheduled timeout: ${nextTimeoutDelay}`)
    this.nextTimeout = setTimeout(() => {
      this.log('queue timeout, *** dispatching ***')
      this.dispatch(true)
    }, nextTimeoutDelay)
  }
  // dispatch up to maximum buffer size requests
  dispatch(timeout) {
    this.smartMonitor.recordSchedule(timeout)
    const queueLength = this.getQueueLength()
    const dispatchLength = Math.min(queueLength, this.maxBufferSize)

    // pop requests from the buffer and update the buffer
    let sendBuffer = this.requestBuffer.splice(0, dispatchLength)

    const sendBufferIds = sendBuffer.map((v) => v.req.id).join(',')
    this.log(`dispatching ids: ${sendBufferIds}`)

    // send the request and respond to the requests
    sendBufferRequest(this.upstreamUrl, sendBuffer, this.isTFServing, (m) => this.log(m), this.smartMonitor)

    // reschedule next timeout
    this.scheduleNextTimeout()
  }
}

const sendBufferRequest = async (upstreamUrl, sendBuffer, isTFServing, logFunc, smartMonitor) => {
  let sendData = sendBuffer.map((v) => v.requestBody)
  // shape the correct request to be sent
  if (isTFServing) {
    sendData = {
      instances: sendData,
    }
  }

  try {
    logFunc(`[FETCH] Sending request ${JSON.stringify(sendData)}`)
    smartMonitor.recordDispatch(sendBuffer.length)
    const requestAt = Date.now()
    const response = await axios.post(upstreamUrl, sendData)
    const responseAt = Date.now()
    const upstreamResponseTime = responseAt - requestAt
    let data;
    if (isTFServing) {
      data = response.data.predictions
    } else {
      data = response.data
    }
    logFunc(`[FETCH] Received response ${JSON.stringify(data)}`)

    // record dispatch results response time
    smartMonitor.recordUpstreamResult(sendBuffer.length, upstreamResponseTime)

    for (let i = 0; i < sendBuffer.length; i++) {
      const req = sendBuffer[i].req
      const responseTime = responseAt - req.receivedAt
      req.respHeader[headerPrefix + 'responseAt'] = responseAt
      req.respHeader[headerPrefix + 'upstreamResponseTime'] = upstreamResponseTime
      req.respHeader[headerPrefix + 'upstreamRequestCount'] = sendBuffer.length
      req.respHeader[headerPrefix + 'responseTime'] = responseTime
      req.respHeader[headerPrefix + 'queueTime'] = requestAt - req.receivedAt
      smartMonitor.recordRresponseTime(responseTime)

      // setting the headers and sending the results
      let response = [data[i]]
      if (isTFServing) {
        response = {
          predictions: [data[i]],
        }
      }
      sendBuffer[i].res.set(sendBuffer[i].req.respHeader).send(response)

      // record departure
      smartMonitor.recordDeparture()
    }
  } catch (error) {
    logger.log('error', `[FETCH] error with upstream request: ${error}`)

    sendBuffer.forEach((v) => {
      v.res.status(500).send({
        error: error.message
      })
    })

    // record departure
    smartMonitor.recordError()

    throw error
  }

}

module.exports = SmartProxy