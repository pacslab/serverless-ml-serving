// source: https://blog.codewithdan.com/using-the-kubernetes-javascript-client-library/
// source2: https://learnk8s.io/real-time-dashboard
// source3: https://github.com/kubernetes-client/javascript/blob/master/examples/cache-example.js
// source4: https://github.com/kubernetes-client/javascript/blob/master/examples/typescript/watch/watch-example.ts

const logger = require(__basedir + '/helpers/logger')

const k8s = require('@kubernetes/client-node')

const kc = new k8s.KubeConfig()
kc.loadFromDefault()

const appsV1Api = kc.makeApiClient(k8s.AppsV1Api)
// const coreV1Api = kc.makeApiClient(k8s.CoreV1Api)

const knative_regex = /(.*)-(\d{5})-deployment/

const liveKnativeDeployments = {}

async function getKnativeDeployments() {
  const deploymentsRes = await appsV1Api.listDeploymentForAllNamespaces()
  let deployments = [];
  console.log(deploymentsRes)
  for (const deployment of deploymentsRes.body.items) {
    let name = deployment.metadata.name
    if (deployment.status.replicas) {
      // add if proper knative deployment
      if (knative_regex.test(name)) {
        const [_, serviceName, versionNumber] = knative_regex.exec(name)
        deployments.push({
          name,
          serviceName,
          versionNumber,
          status: deployment.status.conditions[0].status,
          image: deployment.spec.template.spec.containers[0].image,
          replicas: deployment.status.replicas,
          ports: [],
          services: []
        });
      }
    }
  }
  console.log(deployments)
}
// getKnativeDeployments().catch(console.error.bind(console))

const getCustomDeploymentObject = (deployment) => {
  let name = deployment.metadata.name
  // add if proper knative deployment
  if (knative_regex.test(name)) {
    let [_, serviceName, versionNumber] = knative_regex.exec(name)
    versionNumber = Number(versionNumber)
    let replicas = deployment.status.replicas
    replicas = replicas ? replicas : 0
    return {
      name,
      serviceName,
      versionNumber,
      status: deployment.status.conditions[0].status,
      image: deployment.spec.template.spec.containers[0].image,
      replicas,
      ports: [],
      services: []
    }
  }

  return null
}

const updateLiveDeployment = (deployment) => {
  const serviceName = deployment.serviceName
  const versionNumber = deployment.versionNumber
  // if a newer version exists, don't update
  if (liveKnativeDeployments[serviceName] && liveKnativeDeployments[serviceName].versionNumber > versionNumber) {
    return
  }

  // if version not older, update it
  liveKnativeDeployments[serviceName] = deployment

  logger.log('info', `[KUBE] Service: ${serviceName}, Replicas: ${liveKnativeDeployments[serviceName].replicas}`)
}

const getLiveKnativeDeploymentStatus = (service_name) => {
  return liveKnativeDeployments[service_name]
}

const listFn = () => appsV1Api.listDeploymentForAllNamespaces()
// const path = '/api/v1/replicationcontrollers'

const informer = k8s.makeInformer(kc, '/apis/apps/v1/deployments', listFn)
informer.on('add', (obj) => {
  const deployment = getCustomDeploymentObject(obj)
  // if knative deployment
  if (deployment != null) {
    updateLiveDeployment(deployment)
  }
});
informer.on('update', (obj) => {
  const deployment = getCustomDeploymentObject(obj)
  // if knative deployment
  if (deployment != null) {
    updateLiveDeployment(deployment)
  }
});
informer.on('delete', (obj) => { 
  console.log(`Deleted: ${obj.metadata?.name}`);
  const deployment = getCustomDeploymentObject(obj)
  // if knative deployment
  if (deployment != null) {
    updateLiveDeployment(deployment)
  }
});
informer.on('error', (err) => {
  console.error(err);
  // Restart informer after 5sec
  setTimeout(() => {
    informer.start();
  }, 5000);
});

informer.start()


module.exports = {
  // getKnativeDeployments,
  getLiveKnativeDeploymentStatus,
}
