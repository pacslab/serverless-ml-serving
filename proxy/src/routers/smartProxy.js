const express = require('express')
const router = new express.Router()

// local imports
const logger = require(__basedir + '/helpers/logger')
const SmartProxy = require(__basedir + '/helpers/SmartProxy')
const SmartMonitor = require(__basedir + '/helpers/SmartMonitor')

// configurations
const config = require(__basedir + '/config')
const {
  workloadConfigs,
} = config


let serviceSmartProxies = {}
for (let serviceName in workloadConfigs) {
  let workloadConfig = workloadConfigs[serviceName]

  const smartMonitor = new SmartMonitor(workloadConfig)
  const smartProxy = new SmartProxy(workloadConfig, smartMonitor)

  serviceSmartProxies[serviceName] = smartProxy
}

router.get('/proxy/test', (req, res) => {
  logger.log('debug', '[PROXY] proxy test')
  res.status(200).send({
    msg: 'This is a proxy test',
  })
})

router.post('/proxy/:serviceName', (req, res) => {
  const serviceName = req.params.serviceName
  const serviceProxy = serviceSmartProxies[serviceName]

  req.receivedAt = Date.now()

  if (serviceProxy) {
    logger.debug('info', `[PROXY] received request for service ${serviceName}`)
    serviceProxy.proxy(req, res)
    // res.status(200).send({msg: 'done!'})
  } else {
    logger.log('info', `[PROXY] service not found: ${serviceName}`)
    res.status(404).send({
      error: `service ${serviceName} not found!`
    })
  }
})

router.post('/proxy-config/:serviceName', (req, res) => {
  const serviceName = req.params.serviceName
  const serviceProxy = serviceSmartProxies[serviceName]

  if (serviceProxy) {
    logger.log('info', `[PROXY-CONFIG] configuring for service ${serviceName}`)
    const newConfig = {
      ...serviceProxy.workloadConfig,
      ...req.body
    }
    serviceProxy.setWorkloadConfig(newConfig)
    res.status(200).send(serviceProxy.workloadConfig)
  } else {
    logger.log('info', `[PROXY-CONFIG] service not found: ${serviceName}`)
    res.status(404).send({
      error: `service ${serviceName} not found!`
    })
  }
})

router.get('/proxy-monitor/:serviceName', (req, res) => {
  const serviceName = req.params.serviceName
  const serviceProxy = serviceSmartProxies[serviceName]

  if (serviceProxy) {
    logger.log('info', `[MONITOR-STAT] received request for service ${serviceName}`)
    res.status(200).send(serviceProxy.smartMonitor.getMonitorStatus())
  } else {
    logger.log('info', `[MONITOR-STAT] service not found: ${serviceName}`)
    res.status(404).send({
      error: `service ${serviceName} not found!`
    })
  }
})

module.exports = router
