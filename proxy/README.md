# Proxy

This proxy will be responsible for tracking and updating the life cycle of the
workloads. In addition, it will generate logs that can be used to extract experimentation
results.

```sh
# sample test workload:
docker run --rm -p 5000:5000 ghcr.io/nimamahmoudi/bentoml-iris-classifier:20210507155435 --workers=2
```
