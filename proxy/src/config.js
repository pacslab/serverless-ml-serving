const KN_DOMAIN = process.env.KN_DOMAIN || 'kn.nima-dev.com'

const workloadConfigs = {
  'bentoml-iris': {
    serviceName: 'bentoml-iris',
    upstreamUrl: `http://bentoml-iris.default.${KN_DOMAIN}/predict`,
    maxBufferTimeoutMs: 500,
    maxBufferSize: 3,
    isTFServing: false,
  },
  'tfserving-resnetv2': {
    serviceName: 'tfserving-resnetv2',
    upstreamUrl: `http://tfserving-resnetv2.default.${KN_DOMAIN}/v1/models/resnet:predict`,
    maxBufferTimeoutMs: 500,
    maxBufferSize: 3,
    isTFServing: true,
  },
  'bentoml-onnx-resnet50': {
    serviceName: 'bentoml-onnx-resnet50',
    upstreamUrl: `http://bentoml-onnx-resnet50.default.${KN_DOMAIN}/predict`,
    maxBufferTimeoutMs: 500,
    maxBufferSize: 3,
    isTFServing: false,
  },
  'tfserving-mobilenetv1': {
    serviceName: 'tfserving-mobilenetv1',
    upstreamUrl: `http://tfserving-mobilenetv1.default.${KN_DOMAIN}/v1/models/mobilenet:predict`,
    maxBufferTimeoutMs: 500,
    maxBufferSize: 3,
    isTFServing: true,
  },
  'bentoml-pytorch-fashion-mnist': {
    serviceName: 'bentoml-pytorch-fashion-mnist',
    upstreamUrl: `http://bentoml-pytorch-fashion-mnist.default.${KN_DOMAIN}/predict`,
    maxBufferTimeoutMs: 500,
    maxBufferSize: 3,
    isTFServing: false,
  },
  'bentoml-keras-toxic-comments': {
    serviceName: 'bentoml-keras-toxic-comments',
    upstreamUrl: `http://bentoml-keras-toxic-comments.default.${KN_DOMAIN}/predict`,
    maxBufferTimeoutMs: 500,
    maxBufferSize: 3,
    isTFServing: false,
  },
}

module.exports = {
  REPORT_INTERVAL: 10000,
  PORT: process.env.PORT || 3000,
  LOG_LEVEL: process.env.LOG_LEVEL || 'debug',
  workloadConfigs,
  KN_DOMAIN,
}

