#
# OtterTune - admin.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib import admin
from django.db.utils import ProgrammingError
from django.utils.html import format_html
from django_db_logger.admin import StatusLogAdmin
from django_db_logger.models import StatusLog
from djcelery import models as djcelery_models

from .models import (BackupData, DBMSCatalog, ExecutionTime,
                     KnobCatalog, KnobData, MetricCatalog,
                     MetricData, PipelineData, PipelineRun,
                     Project, Result, Session, Workload, Hardware,
                     SessionKnob)
from .types import VarType, WorkloadStatusType


class DBMSCatalogAdmin(admin.ModelAdmin):
    pass


class KnobCatalogAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'tunable')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter), 'tunable')
    ordering = ('dbms', '-tunable', 'name')

    def get_form(self, request, obj=None, **kwargs):
        exclude = ['category', 'summary', 'description', 'context']
        if obj:
            if obj.vartype not in (VarType.INTEGER, VarType.REAL):
                exclude += ['minval', 'maxval']
            if obj.vartype != VarType.ENUM:
                exclude.append('enumvals')
        kwargs['exclude'] = exclude
        form = super().get_form(request, obj, **kwargs)
        for field_name in ('minval', 'maxval', 'enumvals'):
            if field_name in form.base_fields:
                form.base_fields[field_name].required = False
        return form


class MetricCatalogAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'metric_type')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter), 'metric_type')
    ordering = ('dbms', 'name')

    def get_form(self, request, obj=None, **kwargs):
        kwargs['exclude'] = ('summary',)
        return super().get_form(request, obj, **kwargs)


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
                   ('session', admin.RelatedOnlyFieldListFilter),
                   'tunable')
    ordering = ('session__dbms', 'session__name', '-tunable', 'knob__name')

    def dbms(self, instance):  # pylint: disable=no-self-use
        return instance.session.dbms


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
    ordering = ('-id', '-start_time')


class WorkloadAdmin(admin.ModelAdmin):
    list_display = ('name', 'dbms', 'project', 'workload_status')
    list_filter = (('dbms', admin.RelatedOnlyFieldListFilter),
                   ('project', admin.RelatedOnlyFieldListFilter))

    def workload_status(self, instance):  # pylint: disable=no-self-use
        if instance.status == WorkloadStatusType.MODIFIED:
            color = 'red'
        elif instance.status == WorkloadStatusType.PROCESSING:
            color = 'orange'
        else:  # instance.status == WorkloadStatusType.PROCESSED
            color = 'green'
        return format_html('<span style="color: {};">{}</span>'.format(
            color, WorkloadStatusType.name(instance.status)))
    workload_status.short_description = 'Status'


class TaskMetaAdmin(admin.ModelAdmin):
    list_display = ('colored_status', 'task_result', 'date_done', 'task_traceback')
    list_display_links = ('colored_status', 'task_result')
    readonly_fields = ('id', 'task_id', 'status', 'result', 'date_done',
                       'traceback', 'hidden', 'meta')
    fields = readonly_fields
    list_filter = ('status',)
    list_per_page = 10
    ordering = ('-date_done',)
    max_field_length = 1000

    @staticmethod
    def color_field(text, status):
        if status == 'SUCCESS':
            color = 'green'
        elif status in ('PENDING', 'RECEIVED', 'STARTED'):
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {};">{}</span>'.format(color, text))

    def format_field(self, field):
        text = str(field) if field else ''
        if len(text) > self.max_field_length:
            text = text[:self.max_field_length] + '...'
        return text

    def colored_status(self, instance):
        return self.color_field(instance.status, instance.status)
    colored_status.short_description = 'Status'

    def task_traceback(self, instance):
        text = self.format_field(instance.traceback)
        return format_html('<pre><code>{}</code></pre>'.format(text))
    task_traceback.short_description = 'Traceback'

    def task_result(self, instance):
        res = self.format_field(instance.result)
        return res
    task_result.short_description = 'Result'


class CustomStatusLogAdmin(StatusLogAdmin):
    list_display = ('logger_name', 'colored_msg', 'traceback', 'create_datetime_format')
    list_display_links = ('logger_name',)
    list_filter = ('logger_name', 'level')


class ExecutionTimeAdmin(admin.ModelAdmin):
    list_display = ('event', 'start_time', 'exec_time', 'result')
    list_filter = ('module', 'function', 'tag')
    ordering = ('-start_time',)

    def exec_time(self, instance):  # pylint: disable=no-self-use
        return '{:.0f} sec'.format(instance.execution_time)
    exec_time.short_description = 'Execution Time'


# Admin classes for website models
admin.site.register(DBMSCatalog, DBMSCatalogAdmin)
admin.site.register(KnobCatalog, KnobCatalogAdmin)
admin.site.register(MetricCatalog, MetricCatalogAdmin)
admin.site.register(Session, SessionAdmin)
admin.site.register(Project, ProjectAdmin)
admin.site.register(KnobData, KnobDataAdmin)
admin.site.register(MetricData, MetricDataAdmin)
admin.site.register(Result, ResultAdmin)
admin.site.register(BackupData, BackupDataAdmin)
admin.site.register(PipelineData, PipelineDataAdmin)
admin.site.register(PipelineRun, PipelineRunAdmin)
admin.site.register(Workload, WorkloadAdmin)
admin.site.register(SessionKnob, SessionKnobAdmin)
admin.site.register(Hardware, HardwareAdmin)
admin.site.register(ExecutionTime, ExecutionTimeAdmin)

# Admin classes for 3rd party models
admin.site.unregister(StatusLog)
admin.site.register(StatusLog, CustomStatusLogAdmin)
admin.site.register(djcelery_models.TaskMeta, TaskMetaAdmin)
