

// local imports
const logger = require(__basedir + '/helpers/logger')

// custom functions
const arraySum = (arr) => (arr.length > 0) ? arr.reduce((a,b) => a + b) : 0

// main class
class SmartMonitor {
  loggerPrefix = '[MONITOR]'
  constructor(workloadConfig) {
    this.setWorkloadConfig(workloadConfig)

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
    const {
      currentConcurrency,
    } = this.getCurrentMonitorStatus()

    let { 
      currentArrivalCount: historyArrivalCounts
    } = this.historyStatus
    // make adjustments if necessary
    historyArrivalCounts = (historyArrivalCounts) ? historyArrivalCounts : []
    // default to 1 to avoid division by zero (numerator will automatically be zero when no data)
    const currentWindowLength = historyArrivalCounts.length ? historyArrivalCounts.length : 1

    const windowRPS = arraySum(historyArrivalCounts) / currentWindowLength

    return {
      currentConcurrency,
      // historyArrivalCounts,
      // how many seconds in a monitoring window
      monitoringWindowLength: this.monitoringWindowLength,
      windowRPS,
    }
  }
  periodInterval() {
    const currentMonitorStatus = this.getCurrentMonitorStatus()
    logger.log('debug', this.loggerPrefix + ' ' + `${JSON.stringify(currentMonitorStatus)}`)

    for (let k in currentMonitorStatus) {
      // make it an array if it isn't already one
      if(!Array.isArray(this.historyStatus[k])) {
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

    // reset counts to count them again
    this.resetCounts()
  }
  resetHistory() {
    this.historyStatus = {}
  }
  resetCounts() {
    logger.log('info', this.loggerPrefix + ' ' + 'resetting current counts')
    this.currentArrivalCount = 0
    this.currentErrorCount = 0
    this.currentDepartureCount = 0
    this.currentConcurrency = 0
    this.currentDispatchCount = 0
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
}

module.exports = SmartMonitor
