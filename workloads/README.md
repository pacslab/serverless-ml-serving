# Workloads

This folder will host the workloads that we will be using for our experiments.

## Instance Tracking

To enable tracking which instance has responded to a particular request, we may
require the workloads to return headers indicating which pod has executed them
and on which node that pod is located.

## Possible Future Workloads

This is a list of sources that I hope to check out for getting sample workloads for our
experiments.

- [Python - BentoML](https://knative.dev/community/samples/serving/machinelearning-python-bentoml/)
- ResNet?
- [Kubeflow Serving Libraries - Seldon, BentoML, NVIDIA Triton, TFServing](https://www.kubeflow.org/docs/external-add-ons/serving/)
- MobileNet?
- TFHub workloads?

The following seem like good additions to the list of workloads in the BentoML gallery:

- Pytorch CIFAR10 Sample
- Keras Toxic Comment Classification
