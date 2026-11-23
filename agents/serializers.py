from rest_framework import serializers


class TaskSerializer(serializers.Serializer):
    task = serializers.CharField(max_length=2000)


class ApproveSerializer(serializers.Serializer):
    run_id = serializers.CharField(max_length=16)
    approved = serializers.BooleanField()
    approver = serializers.CharField(max_length=100)
    role = serializers.ChoiceField(choices=["clerk", "finance_manager", "cfo"], default="finance_manager")
