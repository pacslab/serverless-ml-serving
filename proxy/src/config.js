const workloadConfigs = {
  'bentoml-iris': {
    upstreamUrl: "http://localhost:5000/predict",
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

