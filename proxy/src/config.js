const workloadConfigs = {
  'bentoml-iris': {
    serviceName: 'bentoml-iris',
    upstreamUrl: "http://bentoml-iris.default.kn.nima-dev.com/predict",
    maxBufferTimeoutMs: 1000,
    maxBufferSize: 3,
  },
}

module.exports = {
  REPORT_INTERVAL: 10000,
  PORT: process.env.PORT || 3000,
  LOG_LEVEL: process.env.LOG_LEVEL || 'debug',
  workloadConfigs,
}

