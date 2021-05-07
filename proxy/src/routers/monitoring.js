const express = require('express')
const router = new express.Router()

// local imports
const logger = require(__basedir + '/helpers/logger')

// configurations
const config = require(__basedir + '/config')


router.get('/monitoring/test', (req, res) => {
  logger.log('debug','monitoring test')
  res.status(200).send({
    msg: 'This is monitoring test',
  })
})

module.exports = router
