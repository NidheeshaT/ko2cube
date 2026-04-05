"""
KWOK Adapter — Main interface between Ko2cube and the KWOK cluster.

Usage:
    adapter = KwokAdapter()           # reads KWOK_ENABLED env var
    adapter.submit_pod(job, assignment)
    adapter.delete_pod(job_id)
    adapter.teardown()                # deletes all ko2cube pods

Environment variables:
    KWOK_ENABLED    Set to "true" to enable. Default: "false".
                    If false, all methods are no-ops (simulation still works).
    KWOK_NAMESPACE  Kubernetes namespace to use. Default: "ko2cube".
    KWOK_KUBECONFIG Path to kubeconfig. Default: uses kubectl default (~/.kube/config).
"""

import json
import logging
import os
import subprocess
from typing import Optional

import yaml  # PyYAML — already in requirements.txt

from models import Job, JobAssignment
from server.kwok.templates import build_pod_manifest

log = logging.getLogger("kwok_adapter")


class KwokAdapter:
    """
    Submits and deletes Kubernetes Pods on a KWOK cluster to mirror
    the Ko2cube simulator's scheduling decisions.

    All public methods are safe to call even when KWOK_ENABLED=false —
    they simply become no-ops so the simulation never depends on KWOK being up.
    """

    def __init__(self):
        self.enabled = os.environ.get("KWOK_ENABLED", "false").lower() == "true"
        self.namespace = os.environ.get("KWOK_NAMESPACE", "ko2cube")
        self.kubeconfig = os.environ.get("KWOK_KUBECONFIG", "")

        if self.enabled:
            log.info("KWOK adapter enabled (namespace=%s)", self.namespace)
            self._ensure_namespace()
        else:
            log.info("KWOK adapter disabled — running in pure simulation mode")

    # Public API

    def submit_pod(self, job: Job, assignment: JobAssignment) -> bool:
        """
        Submit a Pod to the KWOK cluster for the given job assignment.
        Returns True on success, False on any error (never raises).
        """
        if not self.enabled:
            return True

        manifest = build_pod_manifest(
            job_id=job.job_id,
            cpu_cores=job.cpu_cores,
            memory_gb=job.memory_gb,
            region=assignment.region or "us-east-1",
            instance_type=assignment.instance_type or "m5.large",
            machine_type=assignment.machine_type or "spot",
            eta_minutes=job.eta_minutes,
            namespace=self.namespace,
        )

        return self._kubectl_apply(manifest)

    def delete_pod(self, job_id: str) -> bool:
        """
        Delete a pod by its Ko2cube job_id label.
        Called when a job completes naturally or is dropped.
        Returns True on success, False on any error (never raises).
        """
        if not self.enabled:
            return True

        pod_name = job_id.lower().replace("_", "-")
        return self._kubectl(["delete", "pod", pod_name,
                              "-n", self.namespace,
                              "--ignore-not-found=true"])

    def teardown(self) -> None:
        """Delete all Ko2cube pods (called on reset to clean up previous episode)."""
        if not self.enabled:
            return

        log.info("KWOK teardown: deleting all ko2cube pods in namespace %s", self.namespace)
        self._kubectl([
            "delete", "pods",
            "-n", self.namespace,
            "-l", "app=ko2cube-job",
            "--ignore-not-found=true",
        ])

    def pod_status(self, job_id: str) -> Optional[str]:
        """
        Return the phase of a pod: Pending | Running | Succeeded | Failed | Unknown.
        Returns None if pod doesn't exist or KWOK is disabled.
        """
        if not self.enabled:
            return None

        pod_name = job_id.lower().replace("_", "-")
        result = self._kubectl_output([
            "get", "pod", pod_name,
            "-n", self.namespace,
            "-o", "jsonpath={.status.phase}",
            "--ignore-not-found=true",
        ])
        return result.strip() or None

    # Private helpers

    def _ensure_namespace(self) -> None:
        """Create the ko2cube namespace if it doesn't exist."""
        self._kubectl([
            "create", "namespace", self.namespace,
            "--dry-run=client", "-o", "yaml",
            "|", "kubectl", "apply", "-f", "-",
        ], use_shell=True)

    def _kubectl_apply(self, manifest: dict) -> bool:
        """Apply a manifest dict via kubectl apply -f -"""
        try:
            yaml_str = yaml.dump(manifest, default_flow_style=False)
            cmd = self._base_cmd() + ["apply", "-f", "-"]
            result = subprocess.run(
                cmd,
                input=yaml_str,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                log.warning("kubectl apply failed: %s", result.stderr.strip())
                return False
            log.debug("kubectl apply: %s", result.stdout.strip())
            return True
        except Exception as exc:
            log.warning("kubectl apply exception: %s", exc)
            return False

    def _kubectl(self, args: list, use_shell: bool = False) -> bool:
        """Run a kubectl command. Returns True on success."""
        try:
            cmd = self._base_cmd() + args
            if use_shell:
                cmd_str = " ".join(cmd)
                result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=10)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                log.debug("kubectl %s: %s", args[0], result.stderr.strip())
            return result.returncode == 0
        except Exception as exc:
            log.warning("kubectl exception: %s", exc)
            return False

    def _kubectl_output(self, args: list) -> str:
        """Run kubectl and return stdout as string."""
        try:
            cmd = self._base_cmd() + args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    def _base_cmd(self) -> list:
        """Base kubectl command. Uses KWOK_KUBECTL env var (full path) if set, else system kubectl."""
        kubectl_bin = os.environ.get("KWOK_KUBECTL", "kubectl")
        cmd = [kubectl_bin]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        return cmd
