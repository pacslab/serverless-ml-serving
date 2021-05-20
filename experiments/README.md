# Experimenter

To run a jupyter notebook while recording the new outputs:

```sh
jupyter nbconvert --to notebook --inplace --execute experiment.ipynb
# or the following if facing the kernel not found error:
# jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=python3 --execute experiment.ipynb
# list of kernels:
#   jupyter kernelspec list
```

To terminate pods that are stuck in `Terminating` state:

```sh
# command:
for p in $(kubectl get pods | grep Terminating | awk '{print $1}'); do kubectl delete pod $p --grace-period=0 --force;done
# or add the following to `crontab -e` to run every minute
* * * * * for p in $(kubectl get pods | grep Terminating | awk '{print $1}'); do kubectl delete pod $p --grace-period=0 --force;done
```
