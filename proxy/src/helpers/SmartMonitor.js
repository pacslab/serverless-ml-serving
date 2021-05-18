

// local imports
const logger = require(__basedir + '/helpers/logger')
const kube = require(__basedir + '/helpers/kube')

// custom functions
const arraySum = (arr) => arr.reduce((a, b) => a + b, 0)
const arrayMean = (arr) => (arraySum(arr) / arr.length)
// can be used with .filter to get only unique values
const onlyUnique = (value, index, self) => self.indexOf(value) === index
// get array quantiles
const arrayQuantile = (sorted, q) => {
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  } else {
    return sorted[base];
  }
}
// sort in ascending order
const asc = arr => arr.sort((a, b) => a - b)

// response time stats
const getResponseTimeStats = (arr) => {
  const sorted = asc(arr)
  const q50 = arr.length ? arrayQuantile(sorted, 0.50) : null
  const q95 = arr.length ? arrayQuantile(sorted, 0.95) : null
  const mean = arr.length ? arrayMean(arr) : null

  return {
    q50,
    q95,
    mean,
    count: arr.length,
  }
}

// main class
class SmartMonitor {
  loggerPrefix = '[MONITOR]'
  constructor(workloadConfig) {
    this.setWorkloadConfig(workloadConfig)

    // counts will reset to zero, but concurrency is continuously being updated
    this.currentConcurrency = 0
    this.resetCounts()
    this.resetHistory()

    // managing monitoring stats
    setInterval(() => this.periodInterval(), this.monitoringPeriodInterval)
  }
  setWorkloadConfig(workloadConfig) {
    this.workloadConfig = workloadConfig
    // setting max buffer size to be reported
    this.maxBufferSize = workloadConfig.maxBufferSize
    // we need super fast access to some workload config
    // this.monitoringPeriodInterval = workloadConfig.monitoringPeriodInterval
    // this.monitoringPeriodCount = workloadConfig.monitoringPeriodCount
    // this.monitoringResponseTimePeriodCount = workloadConfig.monitoringResponseTimePeriodCount

    // how long each monitoring period is
    this.monitoringPeriodInterval = 2000
    // how many periods are considered when calculating values
    this.monitoringPeriodCount = 10
    // how many periods are considered for response time monitoring
    this.monitoringResponseTimePeriodCount = 30

    // the number of seconds in the monitoring window in total
    this.monitoringWindowLength = this.monitoringPeriodInterval * this.monitoringPeriodCount / 1000
  }
  getCurrentMonitorStatus() {
    return {
      currentArrivalCount: this.currentArrivalCount,
      currentDepartureCount: this.currentDepartureCount,
      currentConcurrency: this.currentConcurrency,
      currentErrorCount: this.currentErrorCount,
      currentDispatchCount: this.currentDispatchCount,
      currentDispatchRequestCount: this.currentDispatchRequestCount,
      currentMaxBufferSize: this.maxBufferSize,
      currentTimeouts: this.currentTimeouts,
      // what ratio of dispatches were due to timeout
      currentTimeoutRatio: (this.currentDispatchRequestCount > 0) ? (this.currentTimeouts / this.currentDispatchRequestCount) : null,
      currentReplicaCount: kube.getLiveKnativeDeploymentStatus(this.workloadConfig.serviceName)?.replicas,
    }
  }
  // returns the current state of the system, including estimated RPS and concurrency
  getMonitorStatus() {
    // get some parameters out of current monitor status
    const currentMonitorStatus = this.getCurrentMonitorStatus()

    const windowKeys = [
      'currentArrivalCount',
      'currentDepartureCount',
      'currentConcurrency',
      'currentErrorCount',
      'currentDispatchCount',
      'currentReplicaCount',
      'currentMaxBufferSize',
      'currentTimeouts',
    ]
    const windowedHistoryValues = {}
    for (let k of windowKeys) {
      // create the new key to be used
      let newK = k.replace('current', '').replace('Count', '')
      newK = newK.charAt(0).toLowerCase() + newK.slice(1)

      let historyCounts = this.historyStatus[k]
      // refine arrays to avoid undefined
      historyCounts = (historyCounts) ? historyCounts : []
      // calculate rates
      let currentWindowLength = (historyCounts.length > 0) ? historyCounts.length : 1
      windowedHistoryValues[newK] = {
        average: arraySum(historyCounts) / currentWindowLength,
        rate: arraySum(historyCounts) / currentWindowLength / (this.monitoringPeriodInterval / 1000),
      }
    }

    // average ratio of dispatches that were due to timeout
    windowedHistoryValues['timeoutRatio'] = {
      // average: windowedHistoryValues['timeouts'].average / windowedHistoryValues['dispatch'].average,
      average: this.historyStatus['currentTimeouts'] ? arraySum(this.historyStatus['currentTimeouts']) / arraySum(this.historyStatus['currentDispatchRequestCount']) : null,
      rate: -1,
    }

    const windowedUpstreamResponseTimesHistory = this.historyUpstreamResponseTimes.reduce((acc, curr) => acc.concat(curr), [])

    const windowedUpstreamResponseTime = {}
    windowedUpstreamResponseTimesHistory.forEach((v) => {
      // empty array if not defined
      if (windowedUpstreamResponseTime[v[0]] === undefined) {
        windowedUpstreamResponseTime[v[0]] = {
          values: [],
        }
      }
      windowedUpstreamResponseTime[v[0]]['values'].push(v[1])
    })
    for (let k in windowedUpstreamResponseTime) {
      const wrt = windowedUpstreamResponseTime[k]
      const values = wrt['values']
      const WRTStats = getResponseTimeStats(values)
      wrt['stats'] = WRTStats
      wrt['batchSize'] = Number(k)
    }

    // get a list of unique batch sizes
    let windowedUpstreamResponseBatchSizes = asc(Object.keys(windowedUpstreamResponseTime).map(v => Number(v)))
    // used in calculating average
    let batchSizeSum = 0;
    let batchSizeCount = 0;
    Object.keys(windowedUpstreamResponseTime).forEach(v => {
      batchSizeSum += (Number(v) * windowedUpstreamResponseTime[v].stats.count)
      batchSizeCount += windowedUpstreamResponseTime[v].stats.count
    })

    // return downstream response time stats
    const windowedResponseTimesHistory = this.historyResponseTimes.reduce((acc, curr) => acc.concat(curr), [])

    // get the latest stable status object
    const lastMonitorStatus = {}
    for (let k in this.historyStatus) {
      lastMonitorStatus[k] = this.historyStatus[k][this.historyStatus[k].length - 1]
    }

    return {
      // how many seconds in a monitoring window
      monitoringWindowLength: this.monitoringWindowLength,
      monitoringResponseTimeLength: this.monitoringResponseTimePeriodCount * this.monitoringPeriodInterval / 1000,
      monitoringPeriodInterval: this.monitoringPeriodInterval / 1000,
      currentMonitorStatus,
      lastMonitorStatus,
      windowedHistoryValues,
      windowedUpstream: {
        responseTimes: windowedUpstreamResponseTime,
        batchSizes: {
          values: windowedUpstreamResponseBatchSizes,
          average: batchSizeSum / batchSizeCount,
        },
      },
      responseTimes: {
        values: windowedResponseTimesHistory,
        stats: getResponseTimeStats(windowedResponseTimesHistory),
      }
    }
  }
  periodInterval() {
    const currentMonitorStatus = this.getCurrentMonitorStatus()
    // logger.log('debug', this.loggerPrefix + ' ' + `${JSON.stringify(currentMonitorStatus)}`)

    for (let k in currentMonitorStatus) {
      // make it an array if it isn't already one
      if (!Array.isArray(this.historyStatus[k])) {
        this.historyStatus[k] = []
      }

      this.historyStatus[k].push(currentMonitorStatus[k])
      if (this.historyStatus[k].length > this.monitoringPeriodCount) {
        // pop one from the other side of the array if array full
        this.historyStatus[k].shift()
      }

      // for debug purposes, log the history status
      // console.log(k)
      // console.log(this.historyStatus[k])
    }

    this.historyUpstreamResponseTimes.push(this.currentUpstreamResponseTimes)
    if (this.historyUpstreamResponseTimes.length > this.monitoringResponseTimePeriodCount) {
      // pop one from the other side of the array if array full
      this.historyUpstreamResponseTimes.shift()
    }

    this.historyResponseTimes.push(this.currentResponseTimes)
    if (this.historyResponseTimes.length > this.monitoringResponseTimePeriodCount) {
      // pop one from the other side of the array if array full
      this.historyResponseTimes.shift()
    }

    // reset counts to count them again
    this.resetCounts()
  }
  resetHistory() {
    this.historyStatus = {}
    this.historyUpstreamResponseTimes = []
    this.historyResponseTimes = []
  }
  resetCounts() {
    // logger.log('info', this.loggerPrefix + ' ' + 'resetting current counts')
    this.currentArrivalCount = 0
    this.currentErrorCount = 0
    this.currentDepartureCount = 0
    this.currentDispatchCount = 0
    this.currentDispatchRequestCount = 0
    this.currentTimeouts = 0

    // response time recording
    this.currentUpstreamResponseTimes = []
    this.currentResponseTimes = []
  }
  recordArrival() {
    this.currentArrivalCount++
    this.currentConcurrency++
  }
  recordDeparture() {
    this.currentDepartureCount++
    this.currentConcurrency--
  }
  recordError() {
    this.currentErrorCount++
    this.currentConcurrency--
  }
  // record how many inferences have been dispatched
  recordDispatch(count) {
    this.currentDispatchCount += count
    this.currentDispatchRequestCount ++
  }
  recordUpstreamResult(count, responseTime) {
    this.currentUpstreamResponseTimes.push([count, responseTime])
  }
  recordSchedule(timeout) {
    if (timeout) {
      this.currentTimeouts++
    }
  }
  recordRresponseTime(responseTime) {
    this.currentResponseTimes.push(responseTime)
  }
}

module.exports = SmartMonitor
