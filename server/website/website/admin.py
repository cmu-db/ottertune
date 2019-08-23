#
# OtterTune - admin.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib import admin
from djcelery.models import TaskMeta

from .models import (BackupData, DBMSCatalog, KnobCatalog,
                     KnobData, MetricCatalog, MetricData,
                     PipelineData, PipelineRun, Project,
                     Result, Session, Workload, Hardware,
                     SessionKnob)


class BaseAdmin(admin.ModelAdmin):

    @staticmethod
    def dbms_info(obj):
        try:
            return obj.dbms.full_name
        except AttributeError:
            return obj.full_name


class DBMSCatalogAdmin(BaseAdmin):
    list_display = ['dbms_info']


class KnobCatalogAdmin(BaseAdmin):
    list_display = ['name', 'dbms_info', 'tunable']
    ordering = ['name', 'dbms__type', 'dbms__version']
    list_filter = ['tunable']


class MetricCatalogAdmin(BaseAdmin):
    list_display = ['name', 'dbms_info', 'metric_type']
    ordering = ['name', 'dbms__type', 'dbms__version']
    list_filter = ['metric_type']


class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'last_update', 'creation_time')
    fields = ['name', 'user', 'last_update', 'creation_time']


class SessionAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'last_update', 'creation_time')
    list_display_links = ('name',)


class SessionKnobAdmin(admin.ModelAdmin):
    list_display = ('knob', 'session', 'minval', 'maxval', 'tunable')


class HardwareAdmin(admin.ModelAdmin):
    list_display = ('cpu', 'memory', 'storage')


class KnobDataAdmin(BaseAdmin):
    list_display = ['name', 'dbms_info', 'creation_time']
    fields = ['session', 'name', 'creation_time',
              'knobs', 'data', 'dbms']


class MetricDataAdmin(BaseAdmin):
    list_display = ['name', 'dbms_info', 'creation_time']
    fields = ['session', 'name', 'creation_time',
              'metrics', 'data', 'dbms']


class TaskMetaAdmin(admin.ModelAdmin):
    list_display = ['id', 'status', 'date_done']


class ResultAdmin(BaseAdmin):
    list_display = ['result_id', 'dbms_info', 'workload', 'creation_time',
                    'observation_time']
    list_filter = ['dbms__type', 'dbms__version']
    ordering = ['id']

    @staticmethod
    def result_id(obj):
        return obj.id

    @staticmethod
    def workload(obj):
        return obj.workload.name


class BackupDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'result_id']

    @staticmethod
    def result_id(obj):
        return obj.id


class PipelineDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'version', 'task_type', 'workload',
                    'creation_time']
    ordering = ['-creation_time']

    @staticmethod
    def version(obj):
        return obj.pipeline_run.id


class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ['id', 'start_time', 'end_time']


class PipelineResultAdmin(BaseAdmin):
    list_display = ['task_type', 'dbms_info',
                    'hardware_info', 'creation_timestamp']

    @staticmethod
    def hardware_info(obj):
        return obj.hardware.name


class WorkloadAdmin(admin.ModelAdmin):
    list_display = ['workload_id', 'name']

    @staticmethod
    def workload_id(obj):
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
admin.site.register(SessionKnob, SessionKnobAdmin)
admin.site.register(Hardware, HardwareAdmin)
