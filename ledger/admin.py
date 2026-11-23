from django.contrib import admin

from .models import AuditLog, Invoice, Memory, Payment

admin.site.site_header = "FinMind administration"
admin.site.site_title = "FinMind admin"


class ReadOnlyAdmin(admin.ModelAdmin):
    """Payments and audit entries are only ever created by the approved agent workflow."""

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Invoice)
class InvoiceAdmin(admin.ModelAdmin):
    list_display = ("id", "vendor", "category", "amount", "due_date", "status")
    list_filter = ("status", "category", "vendor")
    search_fields = ("id", "vendor", "description")
    date_hierarchy = "due_date"


@admin.register(Payment)
class PaymentAdmin(ReadOnlyAdmin):
    list_display = ("invoice", "amount", "paid_at", "approved_by")
    search_fields = ("invoice__id", "approved_by")


@admin.register(AuditLog)
class AuditLogAdmin(ReadOnlyAdmin):
    list_display = ("ts", "actor", "action", "detail")
    list_filter = ("action",)
    search_fields = ("actor", "detail")


@admin.register(Memory)
class MemoryAdmin(admin.ModelAdmin):
    list_display = ("key", "value")
