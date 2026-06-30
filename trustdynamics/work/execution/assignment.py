from trustdynamics.work.project.project import Project


class AssignmentMixin:
    project: Project
    workers: set[int]
    
    def eligible_workers(self, task_name: str) -> set[int]:
        policy = self.project.assignment_policy.get(task_name)

        if policy is None:
            return set(self.workers)

        if policy.workers is None:
            workers = set(self.workers)
        else:
            workers = set(policy.workers)

        return (workers - policy.exclude_workers) & self.workers