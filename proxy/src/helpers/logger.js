const winston = require('winston')

// configurations
const config = require(__basedir + '/config')
const {
  LOG_LEVEL: log_level,
} = config

let logger;

const getLoggerInstance = () => {
  if (!logger) {
    console.log('initializing logger with log level', log_level)
    winston.level = log_level
    const myformat = winston.format.combine(
      winston.format.colorize(),
      winston.format.timestamp(),
      winston.format.align(),
      winston.format.printf(info => `${info.timestamp} ${info.level}: ${info.message}`)
    );
    const consoleTransport = new winston.transports.Console({
      format: myformat
    })
    const myWinstonOptions = {
      transports: [consoleTransport],
      level: log_level,
    }
    logger = new winston.createLogger(myWinstonOptions)
  }

  return logger
}


module.exports = getLoggerInstance()
