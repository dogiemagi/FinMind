import uuid

from django.db import transaction
from rest_framework.response import Response
from rest_framework.views import APIView

from assistant.services import get_system
from finmind.guardrails import check_input
from finmind.smalltalk import smalltalk

from .models import AgentRun
from .serializers import ApproveSerializer, TaskSerializer


class RunView(APIView):
    """Start Maya on a task. Read-only tasks finish immediately; payment tasks pause for approval
    and are stored as an `AgentRun`."""

    def post(self, request):
        data = TaskSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        task, _ = check_input(data.validated_data["task"])
        chat = smalltalk(task)
        if chat is not None:
            return Response({"status": "done", "answer": chat, "log": [], "kind": "chat"})
        state = get_system().agent.start(task)
        if state["status"] == "awaiting_approval":
            run_id = uuid.uuid4().hex[:8]
            AgentRun.objects.create(run_id=run_id, task=task, state=state)
            return Response({"run_id": run_id, "status": state["status"], "needs_approval": state["__interrupt__"],
                             "log": state["log"]})
        return Response({"status": state["status"], "answer": state.get("answer"), "log": state["log"]})


class ApproveView(APIView):
    """Resume a paused run with a human decision. Policy limits by role are enforced by the agent."""

    def post(self, request):
        data = ApproveSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        d = data.validated_data
        with transaction.atomic():
            run = AgentRun.objects.select_for_update().filter(run_id=d["run_id"], status=AgentRun.Status.AWAITING).first()
            if run is None:
                return Response({"detail": "unknown or already-decided run_id"}, status=404)
            out = get_system().agent.approve(run.state, d["approved"], d["approver"], d["role"])
            run.state = out
            run.status = AgentRun.Status.APPROVED if d["approved"] else AgentRun.Status.REJECTED
            run.save()
        return Response({"status": out["status"], "answer": out.get("answer"), "executed": out.get("executed"),
                         "skipped": out.get("skipped")})
