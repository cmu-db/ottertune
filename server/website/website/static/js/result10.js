var ResultTimeline = (function(window){

// Localize globals
var readCheckbox = window.readCheckbox, getLoadText = window.getLoadText, valueOrDefault = window.valueOrDefault;

function renderPlot(data, div_id) {
    var plotdata = [], series = [];

    plotdata.push(data.data[defaults.result]);
    series.push({"label":  "Result #" + defaults.result});

    $("input[name^='same_run']:checked").each(function() {
        var pk = $(this).val();
        series.push({"label":  "Result #" + pk});
        plotdata.push(data.data[pk]);
    });

    var plotoptions = {
        title: {text: data.metric + " " + data.lessisbetter, fontSize: '1.1em'},
        series: series,
        axes:{
            yaxis:{
                label: data.units,
                labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
                autoscale: true,
                min: 0,
                tickOptions: {formatString:'%.1f'},
            },
            xaxis:{
                label: 'Time (seconds)',
                autoscale: true,
                min: 0,
            }
        },
        legend: {show: true},
        highlighter: {
            show: true,
            tooltipLocation: 'nw',
            yvalues: 2,
            formatString:'<table class="jqplot-highlighter"><tr><td>time:</td><td>%s</td></tr> <tr><td>' + data.metric + ':</td><td>%s</td></tr></table>'
        },
        cursor: {show: true, zoom:true, showTooltip:false, clickReset:true}
    };

    $("#" + div_id).html('<div id="' + div_id + '_plot"></div><div id="plotdescription"></div>');
    $.jqplot(div_id + '_plot', plotdata, plotoptions);
}

function render() {
    $("#plotgrid").html("");
    // render single plot when one benchmark is selected
    $("input[name^='metric']:checked").each(function() {
        var metric = $(this).val();
        var plotid = "plot_" + metric;
        $("#plotgrid").append('<div id="' + plotid + '" class="plotcontainer"></div>');
        renderPlot(defaults.data[metric], plotid);
    });
}

function getConfiguration() {
    var config = {
        id: defaults.result,
        met: readCheckbox("input[name='metric']:checked"),
        same: readCheckbox("input[name='same_run']:checked"),
    };
    return config;
}

function updateUrl() {
    var cfg = getConfiguration();
    $.address.autoUpdate(false);
    for (var param in cfg) {
        $.address.parameter(param, cfg[param]);
    }
  $.address.update();

  
  new_id = $(this).attr("value")
  $.ajaxSettings.async = false;
    $.getJSON('/ajax_new/', {new_id: new_id} , function(ret){
        $.each(defaults.all_metrics, function(){
           m = this;
           defaults.data[this]['data'][new_id] = []
           $.each(ret[m],function(i,value){
           defaults.data[m]['data'][new_id].push(value)})

       })
    }
   )
  
   refreshContent();
}

function updateUrl_metric() {
    var cfg = getConfiguration();
    $.address.autoUpdate(false);
    for (var param in cfg) {
        $.address.parameter(param, cfg[param]);
    }
  $.address.update();
 refreshContent();

}


function refreshContent() {


   render();
}

function initializeSite(event) {
    setValuesOfInputFields(event);
    $("input[name='metric']"  ).on('click', updateUrl_metric);
    $("input[name='same_run']").on('click', updateUrl);
     refreshContent();
}

function refreshSite(event) {
    setValuesOfInputFields(event);
    refreshContent();
}

function setValuesOfInputFields(event) {
    // Either set the default value, or the one parsed from the url

    // Set default selected metrics
    $("input:checkbox[name='metric']").prop('checked', false);
    var metrics = event.parameters.met ? event.parameters.met.split(',') : defaults.metrics;
    $("input:checkbox[name='metric']").each(function() {
        if ($.inArray($(this).val(), metrics) >= 0) {
            $(this).prop('checked', true);
        }
    });

    $("input:checkbox[name='same_run']").prop('checked', false);
    var others = event.parameters.same ? event.parameters.same.split(',') : ["none"];
    $("input:checkbox[name='same_run']").each(function() {
        if ($.inArray($(this).val(), others) >= 0) {
            $(this).prop('checked', true);
        }
    });
}

function init(def) {
    defaults = def;

$.address.init(initializeSite)
}

return {
    init: init
};

})(window);
