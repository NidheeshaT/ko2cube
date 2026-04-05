"""
tests/kwok/test_job.py

Integration tests for Job scheduling and completion.
"""

import pytest

pytestmark = pytest.mark.integration


class TestJobIntegration:

    def test_job_is_created_in_cluster(self, kwok_cluster):
        from server.kwok.job import Job
        ns = kwok_cluster["namespace"]
        job = Job(name="int-job-basic", node_name=kwok_cluster["node_name"],
                  active_deadline_seconds=3600, cluster_name=kwok_cluster["cluster_name"])
        job.create()

        cluster = kwok_cluster["cluster"]
        
        jobs = cluster.get_jobs(ns)
        assert any(j["name"] == "int-job-basic" for j in jobs)
        
        job.delete()

    def test_job_completions_and_parallelism_set(self, kwok_cluster):
        from server.kwok.job import Job
        ns = kwok_cluster["namespace"]
        job = Job(name="int-job-parallel", node_name=kwok_cluster["node_name"],
                  completions=4, parallelism=2, active_deadline_seconds=3600,
                  cluster_name=kwok_cluster["cluster_name"])
        job.create()

        cluster = kwok_cluster["cluster"]
        job_dict = next(j for j in cluster.get_jobs(ns) if j["name"] == "int-job-parallel")
        
        assert job_dict["completions"] == 4
        assert job_dict["parallelism"] == 2
        job.delete()

