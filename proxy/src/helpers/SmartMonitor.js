

// local imports
const logger = require(__basedir + '/helpers/logger')

class SmartMonitor {
  loggerPrefix = '[MONITOR]'
  constructor(workloadConfig) {
    this.setWorkloadConfig(workloadConfig)

    this.resetCounts()

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
  periodInterval() {
    const currentMonitorStatus = this.getCurrentMonitorStatus()
    logger.log('debug', this.loggerPrefix + ' ' + `${JSON.stringify(currentMonitorStatus)}`)
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
