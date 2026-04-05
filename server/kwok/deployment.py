from kubernetes import client, utils
from .config import load_kwok_kubeconfig


class DeploymentDict(dict):
    """Generates the Kubernetes dictionary representation of an apps/v1 Deployment."""
    def __init__(self, name: str, namespace: str, replicas: int, cpu: str, memory: str, image: str):
        super().__init__({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": namespace,
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": name,
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": name,
                        }
                    },
                    "spec": {
                        "automountServiceAccountToken": False,
                        "containers": [
                            {
                                "name": "fake-app-container",
                                "image": image,
                                "resources": {
                                    "requests": {"cpu": cpu, "memory": memory},
                                    "limits":   {"cpu": cpu, "memory": memory},
                                },
                            }
                        ],
                    },
                },
            },
        })


class Deployment:
    """
    Represents a simulated apps/v1 Deployment on a kwok cluster.
    """
    def __init__(
        self,
        name: str = "kwok-deploy-0",
        namespace: str = "default",
        replicas: int = 1,
        cpu: str = "100m",
        memory: str = "128Mi",
        image: str = "fake-image:v1",
        cluster_name: str = "kwok-cluster",
        on_delete_callback=None,
    ):
        self.name = name
        self.namespace = namespace
        self.replicas = replicas
        self.image = image
        self.cluster_name = cluster_name
        self.on_delete_callback = on_delete_callback

        self._dict = DeploymentDict(name, namespace, replicas, cpu, memory, image)

    @property
    def spec(self) -> dict:
        return self._dict

    def create(self):
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        k8s_client = client.ApiClient()
        print(f"Creating deployment '{self.name}' with {self.replicas} replicas...")
        try:
            utils.create_from_dict(k8s_client, self._dict)
            print(f"Deployment '{self.name}' created successfully!")
        except Exception as e:
            print(f"Failed to create deployment: {e}")
            raise

    def delete(self):
        """Delete this deployment from the kwok cluster."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        apps_v1 = client.AppsV1Api()
        print(f"Deleting deployment '{self.name}' in namespace '{self.namespace}'...")
        try:
            apps_v1.delete_namespaced_deployment(
                self.name,
                self.namespace,
                propagation_policy="Foreground"
            )
            print(f"Deployment '{self.name}' deleted.")
            
            if self.on_delete_callback:
                self.on_delete_callback(self.name)
        except Exception as e:
            print(f"Failed to delete deployment: {e}")
            raise

    def apply_new_rollout(self, new_image: str):
        """Patches the Deployment with a new image, triggering a rolling update."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        apps_v1 = client.AppsV1Api()
        patch = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {"name": "fake-app-container", "image": new_image}
                        ]
                    }
                }
            }
        }
        
        print(f"Patching deployment '{self.name}' with image '{new_image}'...")
        try:
            apps_v1.patch_namespaced_deployment(self.name, self.namespace, patch)
            self.image = new_image
            print(f"Rollout triggered for '{self.name}'.")
        except Exception as e:
            print(f"Failed to patch rollout over deployment: {e}")
            raise
