

// local imports
const logger = require(__basedir + '/helpers/logger')

// custom functions
const arraySum = (arr) => arr.reduce((a, b) => a + b, 0)
const arrayMean = (arr) => (arraySum(arr) / arr.length)
// can be used with .filter to get only unique values
const onlyUnique = (value, index, self) => self.indexOf(value) === index
// get array quantiles
const arrayQuantile = (sorted, q) => {
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  console.log('base', base)
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
    // we need super fast access to some workload config
    // this.monitoringPeriodInterval = workloadConfig.monitoringPeriodInterval
    // this.monitoringPeriodCount = workloadConfig.monitoringPeriodCount

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
    ]
    const windowedHistoryValues = {}
    for (let k of windowKeys) {
      // create the new key to be used
      let newK = k.replace('current', '').replace('Count', '') + 'Average'
      newK = newK.charAt(0).toLowerCase() + newK.slice(1)

      let historyCounts = this.historyStatus[k]
      // refine arrays to avoid undefined
      historyCounts = (historyCounts) ? historyCounts : []
      // calculate rates
      const currentWindowLength = historyCounts.length ? historyCounts.length : 1
      windowedHistoryValues[newK] = arraySum(historyCounts) / currentWindowLength
    }
    for (let k in windowedHistoryValues) {
      const newK = k.replace('Average', 'Rate')
      // skip if it is concurrency
      if (newK.startsWith('concurrency')) continue
      // divide the average by the window time to get the rate
      windowedHistoryValues[newK] = windowedHistoryValues[k] / (this.monitoringPeriodInterval / 1000)
    }

    const windowedUpstreamResponseTimesHistory = this.historyResponseTimes.reduce((acc, curr) => acc.concat(curr), [])

    const windowedUpstreamResponseTime = {}
    windowedUpstreamResponseTimesHistory.forEach((v) => {
      // empty array if not defined
      if (windowedUpstreamResponseTime[v[0]] === undefined) {
        windowedUpstreamResponseTime[v[0]] = {
          'values': [],
        }
      }
      windowedUpstreamResponseTime[v[0]]['values'].push(v[1])
    })
    for(let k in windowedUpstreamResponseTime) {
      const wrt = windowedUpstreamResponseTime[k]
      const values = wrt['values']
      const WRTStats = getResponseTimeStats(values)
      wrt['stats'] = WRTStats
    }

    // get a list of unique batch sizes
    // let windowedUpstreamResponseBatchSizes = windowedUpstreamResponseTimesHistory.map((v) => v[0])
    // windowedUpstreamResponseBatchSizes = windowedUpstreamResponseBatchSizes.filter(onlyUnique)
    let windowedUpstreamResponseBatchSizes = asc(Object.keys(windowedUpstreamResponseTime).map(v => Number(v)))

    return {
      // how many seconds in a monitoring window
      monitoringWindowLength: this.monitoringWindowLength,
      monitoringResponseTimeLength: this.monitoringResponseTimePeriodCount * this.monitoringPeriodInterval / 1000,
      monitoringPeriodInterval: this.monitoringPeriodInterval / 1000,
      currentMonitorStatus,
      windowedHistoryValues,
      windowedUpstream: {
        responseTimes: windowedUpstreamResponseTime,
        batchSizes: windowedUpstreamResponseBatchSizes,
      }
    }
  }
  periodInterval() {
    const currentMonitorStatus = this.getCurrentMonitorStatus()
    logger.log('debug', this.loggerPrefix + ' ' + `${JSON.stringify(currentMonitorStatus)}`)

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
    this.historyResponseTimes = []
  }
  resetCounts() {
    logger.log('info', this.loggerPrefix + ' ' + 'resetting current counts')
    this.currentArrivalCount = 0
    this.currentErrorCount = 0
    this.currentDepartureCount = 0
    this.currentDispatchCount = 0

    // response time recording
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
  }
  recordUpstreamResult(count, responseTime) {
    this.currentResponseTimes.push([count, responseTime])
  }
}

module.exports = SmartMonitor
