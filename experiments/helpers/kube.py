# generam imports
from threading import Thread
import time
import re
from collections import namedtuple

# library imports
from kubernetes import client, config, watch

# my imports
from .util import zero_if_none

config.load_kube_config()

api_instance = client.AppsV1Api()


def get_replica_and_ready(deployment_name, deployment_ns="default"):
    api_response = api_instance.read_namespaced_deployment(
        deployment_name, deployment_ns)
    replicas = api_response.status.replicas
    ready_replicas = api_response.status.ready_replicas

    return zero_if_none(replicas), zero_if_none(ready_replicas)


def set_replica_num(rnum, deployment_name, deployment_ns="default"):
    rnum = int(rnum)
    if rnum < 1:
        rnum = 1
    api_response = api_instance.read_namespaced_deployment(
        deployment_name, deployment_ns)
    api_response.spec.replicas = rnum
    api_instance.patch_namespaced_deployment_scale(
        deployment_name, deployment_ns, api_response)


# watch deployments functionality
wt = None
live_deployments = {}
stop_watch_thread_signal = False
knative_regex = '(.*)-(\d{5})-deployment'
KnativeDeployment = namedtuple(
    'KnativeDeployment', ['deployment', 'name', 'revision', 'revision_num'])

# Here, out goal is to watch the deployments and be notified of any changes in them. For more information,
# visit the docs:
# https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/AppsV1Api.md#list_deployment_for_all_namespaces
# tutorial:
# https://medium.com/@sebgoa/kubernets-async-watches-b8fa8a7ebfd4


def watch_deploymets():
    global live_deployments
    live_deployments = {}
    w = watch.Watch()
    while stop_watch_thread_signal == False:
        for event in w.stream(api_instance.list_deployment_for_all_namespaces, timeout_seconds=10):
            name = event['object'].metadata.name
            kind = event['object'].kind
            replicas = event['object'].status.replicas
            ready_replicas = event['object'].status.ready_replicas

            live_deployments[name] = {
                'kind': kind,
                'replicas': zero_if_none(replicas),
                'ready_replicas': zero_if_none(ready_replicas),
                'last_update': time.time(),
            }

    print('stopping watch thread')


def start_watch_thread():
    global wt
    if wt is None or not(wt.is_alive()):
        print("starting watch thread...")
        wt = Thread(target=watch_deploymets, args=(), daemon=True)
        wt.start()
        # wait for initial results to appear
        time.sleep(1)
    else:
        print('reusing already existing thread')


def stop_watch_thread():
    global wt
    global stop_watch_thread_signal
    while(wt.is_alive()):
        stop_watch_thread_signal = True
        time.sleep(1)
    print('Thread stopped successfully.')


def extract_knative_deployment(d):
    m = re.match(knative_regex, d)
    name = m.group(1)
    revision = m.group(2)
    return KnativeDeployment(deployment=d, name=name, revision=revision, revision_num=int(revision))


def is_knative_deployment(name):
    m = re.match(knative_regex, name)
    if m:
        return True

    return False


def get_knative_deployments():
    global live_deployments
    return [extract_knative_deployment(d) for d in live_deployments if is_knative_deployment(d)]


def get_kn_latest_revs():
    kn_deployments = get_knative_deployments()
    kn_latest_revs = {}
    for k in kn_deployments:
        if k.name not in kn_latest_revs:
            kn_latest_revs[k.name] = k
        else:
            if k.revision_num > kn_latest_revs[k.name].revision_num:
                kn_latest_revs[k.name] = k
    return kn_latest_revs


# get kn cli command for a deployment
def get_kn_command(name, image, env=None, opts=None, annotations=None, sep=" \\\n  ", **kwargs):
    command = f'kn service apply {name} --image {image}'
    if env is not None and len(env) > 0:
        command += sep
        command += sep.join([f"--env {k}={env[k]}" for k in env])
    if opts is not None:
        command += sep
        command += sep.join([f"{k} {opts[k]}" for k in opts])
    if annotations is not None:
        command += sep
        command += sep.join([f"-a {k}={annotations[k]}" for k in annotations])
    return command
