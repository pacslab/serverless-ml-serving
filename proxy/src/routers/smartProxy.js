const express = require('express')
const router = new express.Router()

// local imports
const logger = require(__basedir + '/helpers/logger')

// configurations
const config = require(__basedir + '/config')


router.get('/proxy/test', (req, res) => {
  logger.log('debug','proxy test')
  res.status(200).send({
    msg: 'This is proxy test',
  })
})

module.exports = router
