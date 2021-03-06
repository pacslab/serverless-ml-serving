"use strict"

global.__basedir = __dirname

// local imports
const logger = require(__basedir + '/helpers/logger')

// imports
const express = require('express')
const http = require('http')
const socketio = require('socket.io')
const addRequestId = require('express-request-id')()

// configurations
const config = require(__basedir + '/config')
const {
  PORT: port,
  KN_DOMAIN,
} = config

logger.info(`starting application on port ${port}`)
logger.log('info', `[CONFIG] Knative Domain: ${KN_DOMAIN}`)

// preparing express app
const app = express()
const server = http.createServer(app)
const io = socketio(server)
app.io = io
global.app = app

// Parse incoming json
app.use(express.json({ limit: '50mb' }))
// Parse form data
app.use(express.urlencoded({ extended: true, limit: '50mb' }))
// CORS
const cors = require('cors')
app.use(cors())
// request id
app.use(addRequestId)


// add routers
const monitoringRouter = require('./routers/monitoring')
app.use(monitoringRouter)
const smartProxyRouter = require('./routers/smartProxy')
app.use(smartProxyRouter)

// Home Page
app.get('', (req, res) => {
  res.send({
    "msg": "this is a test!",
  })
})

// 404 Page
app.get('*', (req, res) => {
  res.status(404).send({
    "err": "not found!",
  })
})

server.listen(port, () => {
  logger.info(`\nServer is up:\n\n\thttp://localhost:${port}\n\n`)
})
