const KN_DOMAIN = process.env.KN_DOMAIN || 'kn.nima-dev.com'

const workloadConfigs = {
  'bentoml-iris': {
    serviceName: 'bentoml-iris',
    upstreamUrl: `http://bentoml-iris.default.${KN_DOMAIN}/predict`,
    maxBufferTimeoutMs: 1000,
    maxBufferSize: 3,
    isTFServing: false,
  },
  'tfserving-resnetv2': {
    serviceName: 'tfserving-resnetv2',
    upstreamUrl: `http://tfserving-resnetv2.default.${KN_DOMAIN}/v1/models/resnet:predict`,
    maxBufferTimeoutMs: 1000,
    maxBufferSize: 3,
    isTFServing: true,
  },
  'bentoml-onnx-resnet50': {
    serviceName: 'bentoml-onnx-resnet50',
    upstreamUrl: `http://bentoml-onnx-resnet50.default.${KN_DOMAIN}/predict`,
    maxBufferTimeoutMs: 1000,
    maxBufferSize: 3,
    isTFServing: false,
  },
}

module.exports = {
  REPORT_INTERVAL: 10000,
  PORT: process.env.PORT || 3000,
  LOG_LEVEL: process.env.LOG_LEVEL || 'debug',
  workloadConfigs,
}

