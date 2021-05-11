

// local imports
const logger = require(__basedir + '/helpers/logger')

// custom functions
const arraySum = (arr) => (arr.length > 0) ? arr.reduce((a, b) => a + b) : 0
const onlyUnique = (value, index, self) => self.indexOf(value) === index
// sort in ascending order
const asc = arr => arr.sort((a, b) => a - b)

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
      const newK = k.replace('current', '')
      let historyCounts = this.historyStatus[k]
      // refine arrays to avoid undefined
      historyCounts = (historyCounts) ? historyCounts : []
      // calculate rates
      const currentWindowLength = historyCounts.length ? historyCounts.length : 1
      windowedHistoryValues[newK] = arraySum(historyCounts) / currentWindowLength
    }

    const windowedUpstreamResponseTimesHistory = this.historyResponseTimes.reduce((acc, curr) => acc.concat(curr), [])

    const windowedUpstreamResponseTime = {}
    windowedUpstreamResponseTimesHistory.forEach((v) => {
      // empty array if not defined
      if(windowedUpstreamResponseTime[v[0]] === undefined) {
        windowedUpstreamResponseTime[v[0]] = []
      }
      windowedUpstreamResponseTime[v[0]].push(v[1]) 
    })

    // get a list of unique batch sizes
    // let windowedUpstreamResponseBatchSizes = windowedUpstreamResponseTimesHistory.map((v) => v[0])
    // windowedUpstreamResponseBatchSizes = windowedUpstreamResponseBatchSizes.filter(onlyUnique)
    let windowedUpstreamResponseBatchSizes = asc(Object.keys(windowedUpstreamResponseTime).map(v => Number(v)))

    return {
      // how many seconds in a monitoring window
      monitoringWindowLength: this.monitoringWindowLength,
      monitoringResponseTimeLength: this.monitoringResponseTimePeriodCount * this.monitoringPeriodInterval / 1000,
      currentMonitorStatus,
      windowedHistoryValues,
      windowedUpstreamResponseTime,
      windowedUpstreamResponseBatchSizes,
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
