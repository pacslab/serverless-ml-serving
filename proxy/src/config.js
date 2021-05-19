const workloadConfigs = {
  'bentoml-iris': {
    serviceName: 'bentoml-iris',
    upstreamUrl: "http://bentoml-iris.default.kn.nima-dev.com/predict",
    maxBufferTimeoutMs: 1000,
    maxBufferSize: 3,
    isTFServing: false,
  },
  'tfserving-resnetv2': {
    serviceName: 'bentoml-iris',
    upstreamUrl: "http://tfserving-resnetv2.default.kn.nima-dev.com/v1/models/resnet:predict",
    maxBufferTimeoutMs: 1000,
    maxBufferSize: 3,
    isTFServing: true,
  },
}

module.exports = {
  REPORT_INTERVAL: 10000,
  PORT: process.env.PORT || 3000,
  LOG_LEVEL: process.env.LOG_LEVEL || 'debug',
  workloadConfigs,
}

