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


class DBMSCatalogAdmin(admin.ModelAdmin):
    pass


class KnobCatalogAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'tunable')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter), 'tunable')
    ordering = ('dbms', '-tunable', 'name')


class MetricCatalogAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'metric_type')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter), 'metric_type')
    ordering = ('dbms', 'name')


class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'last_update', 'creation_time')
    list_filter = (('user', admin.RelatedOnlyFieldListFilter),)
    ordering = ('name', 'user__username')


class SessionAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'project', 'last_update', 'creation_time')
    list_filter = (('user', admin.RelatedOnlyFieldListFilter),
                   ('project', admin.RelatedOnlyFieldListFilter))
    ordering = ('name', 'user__username', 'project__name')


class SessionKnobAdmin(admin.ModelAdmin):
    list_display = ('knob', 'dbms', 'session', 'minval', 'maxval', 'tunable')
    list_filter = (('session__dbms', admin.RelatedOnlyFieldListFilter),
                   ('session', admin.RelatedOnlyFieldListFilter), ('tunable'))
    ordering = ('session__dbms', 'session__name', '-tunable', 'knob__name')

    @staticmethod
    def dbms(obj):
        return obj.session.dbms


class HardwareAdmin(admin.ModelAdmin):
    pass


class KnobDataAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'session', 'creation_time')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter),
                   ('session', admin.RelatedOnlyFieldListFilter))
    ordering = ('creation_time',)


class MetricDataAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'session', 'creation_time')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter),
                   ('session', admin.RelatedOnlyFieldListFilter))
    ordering = ('creation_time',)


class TaskMetaAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'task_result', 'date_done')
    readonly_fields = ('id', 'task_id', 'status', 'result', 'date_done',
                       'traceback', 'hidden', 'meta')
    fields = readonly_fields
    list_filter = ('status',)
    ordering = ('date_done',)

    @staticmethod
    def task_result(obj, maxlen=300):
        res = obj.result
        if res and len(res) > maxlen:
            res = res[:maxlen] + '...'
        return res


class ResultAdmin(admin.ModelAdmin):
    readonly_fields = ('dbms', 'knob_data', 'metric_data', 'session', 'workload')
    list_display = ('id', 'dbms', 'session', 'workload', 'creation_time')
    list_select_related = ('dbms', 'session', 'workload')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter),
                   ('session', admin.RelatedOnlyFieldListFilter),
                   ('workload', admin.RelatedOnlyFieldListFilter))
    ordering = ('creation_time',)


class BackupDataAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'result')
    ordering = ('id',)


class PipelineDataAdmin(admin.ModelAdmin):
    readonly_fields = ('pipeline_run',)
    list_display = ('id', 'pipeline_run', 'task_type', 'workload', 'creation_time')
    list_filter = ('task_type', ('workload', admin.RelatedOnlyFieldListFilter))
    ordering = ('pipeline_run', 'creation_time')


class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ('id', 'start_time', 'end_time')
    ordering = ('id', 'start_time')


class WorkloadAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'hardware')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter),
                   ('hardware', admin.RelatedOnlyFieldListFilter))


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
