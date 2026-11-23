from django.db import models


class AgentRun(models.Model):
    """A paused agent run. Persisting the graph state in the database (instead of process memory)
    means an approval survives restarts and works behind multiple workers."""

    class Status(models.TextChoices):
        AWAITING = "awaiting_approval"
        APPROVED = "approved"
        REJECTED = "rejected"

    run_id = models.CharField(max_length=16, primary_key=True)
    task = models.TextField()
    state = models.JSONField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.AWAITING, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.run_id} · {self.status}"
