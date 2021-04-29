# BentoML Iris Model

source: https://knative.dev/community/samples/serving/machinelearning-python-bentoml/
source2: https://www.kubeflow.org/docs/external-add-ons/serving/bentoml/
source3: https://docs.bentoml.org/en/latest/quickstart.html


This workload resempbles simpler machine learning tasks.

sample curl for knative:

```sh
$ curl -v -i \
    --header "Content-Type: application/json" \
    --header "Host: iris-classifier.bentoml.example.com" \
    --request POST \
    --data '[[5.1, 3.5, 1.4, 0.2]]' \
    http://192.168.64.4:31871/predict
```

delete all bentoml models:

```sh
kubectl delete namespace bentoml
```
