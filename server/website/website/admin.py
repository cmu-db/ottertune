from django.contrib import admin
from djcelery.models import TaskMeta

from .models import (BackupData, DBMSCatalog, KnobCatalog,
                     KnobData, MetricCatalog, MetricData,
                     PipelineData, PipelineRun, Project,
                     Result, Session, Workload)


class DBMSCatalogAdmin(admin.ModelAdmin):
    list_display = ['dbms_info']

    def dbms_info(self, obj):
        return obj.full_name


class KnobCatalogAdmin(admin.ModelAdmin):
    list_display = ['name', 'dbms_info', 'tunable']
    ordering = ['name', 'dbms__type', 'dbms__version']
    list_filter = ['tunable']

    def dbms_info(self, obj):
        return obj.dbms.full_name


class MetricCatalogAdmin(admin.ModelAdmin):
    list_display = ['name', 'dbms_info', 'metric_type']
    ordering = ['name', 'dbms__type', 'dbms__version']
    list_filter = ['metric_type']

    def dbms_info(self, obj):
        return obj.dbms.full_name


class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'last_update', 'creation_time')
    fields = ['name', 'user', 'last_update', 'creation_time']


class SessionAdmin(admin.ModelAdmin):
    fields = ['name', 'user', 'description',
              'creation_time', 'last_update', 'upload_code',
              'nondefault_settings']
    list_display = ('name', 'user', 'last_update', 'creation_time')
    list_display_links = ('name',)


class KnobDataAdmin(admin.ModelAdmin):
    list_display = ['name', 'dbms_info', 'creation_time']
    fields = ['session', 'name', 'creation_time',
              'knobs', 'data', 'dbms']

    def dbms_info(self, obj):
        return obj.dbms.full_name


class MetricDataAdmin(admin.ModelAdmin):
    list_display = ['name', 'dbms_info', 'creation_time']
    fields = ['session', 'name', 'creation_time',
              'metrics', 'data', 'dbms']

    def dbms_info(self, obj):
        return obj.dbms.full_name


class TaskMetaAdmin(admin.ModelAdmin):
    list_display = ['id', 'status', 'date_done']


class ResultAdmin(admin.ModelAdmin):
    list_display = ['result_id', 'dbms_info', 'workload', 'creation_time',
                    'observation_time']
    list_filter = ['dbms__type', 'dbms__version']
    ordering = ['id']

    def result_id(self, obj):
        return obj.id

    def dbms_info(self, obj):
        return obj.dbms.full_name

    def workload(self, obj):
        return obj.workload.name


class BackupDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'result_id']

    def result_id(self, obj):
        return obj.id


class PipelineDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'version', 'task_type', 'workload',
                    'creation_time']
    ordering = ['-creation_time']

    def version(self, obj):
        return obj.pipeline_run.id


class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ['id', 'start_time', 'end_time']


class PipelineResultAdmin(admin.ModelAdmin):
    list_display = ['task_type', 'dbms_info',
                    'hardware_info', 'creation_timestamp']

    def dbms_info(self, obj):
        return obj.dbms.full_name

    def hardware_info(self, obj):
        return obj.hardware.name


class WorkloadAdmin(admin.ModelAdmin):
    list_display = ['workload_id', 'name']

    def workload_id(self, obj):
        return obj.pk


admin.site.register(DBMSCatalog, DBMSCatalogAdmin)
admin.site.register(KnobCatalog, KnobCatalogAdmin)
admin.site.register(MetricCatalog, MetricCatalogAdmin)
admin.site.register(Session, SessionAdmin)
admin.site.register(Project, ProjectAdmin)
admin.site.register(KnobData, KnobDataAdmin)
admin.site.register(MetricData, MetricDataAdmin)
admin.site.register(TaskMeta, TaskMetaAdmin)
admin.site.register(Result, ResultAdmin)
admin.site.register(BackupData, BackupDataAdmin)
admin.site.register(PipelineData, PipelineDataAdmin)
admin.site.register(PipelineRun, PipelineRunAdmin)
admin.site.register(Workload, WorkloadAdmin)
