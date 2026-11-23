from django.contrib import admin

from .models import AgentRun


@admin.register(AgentRun)
class AgentRunAdmin(admin.ModelAdmin):
    list_display = ("run_id", "task", "status", "created_at", "updated_at")
    list_filter = ("status",)
    search_fields = ("run_id", "task")
    readonly_fields = ("run_id", "task", "state", "status", "created_at", "updated_at")

    def has_add_permission(self, request):
        return False
