var Timeline = (function(window){

// Localize globals
var readCheckbox = window.readCheckbox, getLoadText = window.getLoadText, valueOrDefault = window.valueOrDefault;

var baselineColor = "#d8b83f",
    seriesColors = ["#4bb2c5", "#EAA228", "#579575", "#953579", "#839557", "#ff5800", "#958c12", "#4b5de4", "#0085cc"],
    defaults;

function shouldPlotEquidistant() {
    return $("#equidistant").is(':checked');
}

function OnMarkerClickHandler(ev, gridpos, datapos, neighbor, plot) {
    if($("input[name='workload']:checked").val() === "grid") { return false; }
    if (neighbor) {
        result_id = neighbor.data[3];
        window.location = "/projects/" + defaults.project + "/sessions/" + defaults.session + "/results/" + result_id;
    }
}

function renderPlot(data, div_id) {
    var plotdata = [], series = [];

    for (dbms in data.data) {
        series.push({"label":  dbms});
        plotdata.push(data.data[dbms]);
    }

    $("#" + div_id).html('<div id="' + div_id + '_plot"></div><div id="plotdescription"></div>');

    var plotoptions = {
        title: {text: data.print_metric + " " + data.lessisbetter, fontSize: '1.1em'},
        series: series,
        axes:{
        yaxis:{
            label: data.units,
            labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
            min: 0,
            autoscale: true,
        },
        xaxis:{
            renderer: (shouldPlotEquidistant()) ? $.jqplot.CategoryAxisRenderer : $.jqplot.DateAxisRenderer,
            label: 'Date',
            labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
            tickRenderer: $.jqplot.CanvasAxisTickRenderer,
            tickOptions:{formatString:'%#m/%#d %H:%M', angle:-40},
            autoscale: true,
            rendererOptions:{sortMergedLabels:true}
        }
        },
        legend: {show: true, location: 'nw'},
        highlighter: {
            show: true,
            tooltipLocation: 'nw',
            yvalues: 2,
            formatString:'<table class="jqplot-highlighter"><tr><td>date:</td><td>%s</td></tr> <tr><td>result:</td><td>%s</td></tr></table>'
        },
        cursor: {show: true, zoom:true, showTooltip:false, clickReset:true}
    };
    //Render plot
    $.jqplot(div_id + '_plot',  plotdata, plotoptions);
}

function renderKnobPlot(data, div_id) {
    plotdata=[data.data];

    $("#" + div_id).html('<div id="' + div_id + '_plot"></div><div id="plotdescription"></div>');

    var plotoptions = {
        title: {text: data.knob, fontSize: '1.1em'},
        //series: [{"label":"Hello"}],
        axes:{
        yaxis:{
            label: data.units,
            labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
            min: 0,
            autoscale: true,
        },
        xaxis:{
            renderer: (shouldPlotEquidistant()) ? $.jqplot.CategoryAxisRenderer : $.jqplot.DateAxisRenderer,
            label: 'Date',
            labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
            tickRenderer: $.jqplot.CanvasAxisTickRenderer,
            tickOptions:{formatString:'%#m/%#d %H:%M', angle:-40},
            autoscale: true,
            rendererOptions:{sortMergedLabels:true}
        }
        },
        legend: {show: true, location: 'nw'},
        highlighter: {
            show: true,
            tooltipLocation: 'nw',
            yvalues: 2,
            formatString:'<table class="jqplot-highlighter"><tr><td>date:</td><td>%s</td></tr> <tr><td>result:</td><td>%s</td></tr></table>'
        },
        cursor: {show: true, zoom:true, showTooltip:false, clickReset:true}
    };
    //Render plot
    $.jqplot(div_id + '_plot',  plotdata, plotoptions);
}

function renderMiniplot(plotid, data) {
    var plotdata = [], series = [];

    for (dbms in data.data) {
        series.push("");
        plotdata.push(data.data[dbms]);
    }

    var plotoptions = {
        title: {text: data.workload + ": " + data.metric, fontSize: '1.1em'},
        seriesDefaults: {lineWidth: 2, markerOptions:{style:'circle', size: 6}},
        series: series,
        axes: {
        yaxis: {
            min: 0, autoscale:true, showTicks: false
        },
        xaxis: {
            renderer:$.jqplot.DateAxisRenderer,
            pad: 1.01,
            autoscale:true,
            showTicks: false
        }
        },
        highlighter: {show:false},
        cursor:{showTooltip: false, style: 'pointer'}
    };
    $.jqplot(plotid, plotdata, plotoptions);
}

var fixed_header = null;

function gen_url(result_id) {
	return "{% url 'result' project_id session_id result_id %}".replace("project_id", defaults.project_id).replace("session_id", defaults.session_id).replace("result_id", result_id)
}

function render(data) {
    disable_options(false);

    $("#plotgrid").html("");
    if(data.error !== "None") {
        var h = $("#content").height();//get height for error message
        $("#plotgrid").html(getLoadText(data.error, h, false));
        return 1;
    } else if ($("input[name='workload']:checked").val() === "show_none") {
        var h = $("#content").height();//get height for error message
        $("#plotgrid").html(getLoadText("Please select a workload on the left", h, false));
    } else if (data.timelines.length === 0) {
        var h = $("#content").height();//get height for error message
        $("#plotgrid").html(getLoadText("No data available", h, false));
    } else if ($("input[name='workload']:checked").val() === "grid"){
        //Render Grid of plots
        disable_options(true);
        for (var wkld in data.timelines) {
            var plotid = "plot_" + data.timelines[wkld].workload;
            $("#plotgrid").append('<div id="' + plotid + '" class="miniplot"></div>');
            $("#" + plotid).click(function() {
                var wkld = $(this).attr("id").slice(5);
                $("#workload_" + wkld).trigger("click");//.prop("checked", true);
                updateUrl();
            });
            renderMiniplot(plotid, data.timelines[wkld]);
        }
    } else {
        // render single plot when one workload is selected
    	var i = 0;
        for (var metric in data.timelines) {
            var plotid = "plot_" + i;
            $("#plotgrid").append('<div id="' + plotid + '" class="plotcontainer"></div>');
            renderPlot(data.timelines[metric], plotid);
            i = i + 1;
        }
        for (var knob in data.knobtimelines) {
            var plotid = "plot_" + i;
            $("#plotgrid").append('<div id="' + plotid + '" class="plotcontainer"></div>');
            renderKnobPlot(data.knobtimelines[knob], plotid);
            i = i + 1;
        }
    }
    var dt = $("#dataTable").dataTable( {
        "aaData": data.results,
        "aoColumns": [
            { "sTitle": data.columnnames[0], "sClass": "center", "sType": "num-html", "mRender": function (data, type, full) {
            	return '<a href="/projects/' + defaults.project + '/sessions/' + defaults.session + '/results/' + data + '">' + data + '</a>';
            }},
            { "sTitle": data.columnnames[1], "sClass": "center"},
            { "sTitle": data.columnnames[2], "sClass": "center", "mRender": function (data, type, full) {
                return '<a href="/projects/' + defaults.project + '/sessions/' + defaults.session + '/knobs/' + full[7] + '">' + data + '</a>';
            }},
            { "sTitle": data.columnnames[3], "sClass": "center", "mRender": function (data, type, full) {
            	return '<a href="/projects/' + defaults.project + '/sessions/' + defaults.session + '/metrics/' + full[8] + '">' + data + '</a>';
            }},
            { "sTitle": data.columnnames[4], "sClass": "center", "mRender": function (data, type, full) {
                return '<a href="/projects/' + defaults.project + '/sessions/' + defaults.session + '/workloads/' + full[9] + '">' + data + '</a>';
            }},
            { "sTitle": data.columnnames[5], "sClass": "center", "mRender": function (data, type, full) {
            	return data.toFixed(2);
            }},
        ],
        "bFilter": false,
        "bAutoWidth": true,
        "sPaginationType": "full_numbers",
        "bDestroy": true
    });
    
    if (fixed_header != null) {
        fixed_header.fnUpdate();
    } else {
        fixed_header = new FixedHeader(dt);
    }
}

function refreshContent() {
    var h = $("#content").height();//get height for loading text
    $("#plotgrid").fadeOut("fast", function() {
        $("#plotgrid").html(getLoadText("Loading...", h, true)).show();
        $.getJSON("/get_data/", getConfiguration(), render);
    });
}

function updateUrl() {
    var cfg = getConfiguration();
    $.address.autoUpdate(false);
    for (var param in cfg) {
        $.address.parameter(param, cfg[param]);
    }
    $.address.update();
}

function disable_options(value) {
    $("#results_per_page").attr("disabled", value);
    $('#results_per_page').selectpicker('refresh');
    $("#equidistant").attr("disabled", value);
    $("input:checkbox[name='metric']").each(function() {
        $(this).attr('disabled', value);
    });
    $.each(defaults.additional, function(i, add) {
        $("select[name^='additional_" + add + "']").attr('disabled', value);
        $("select[name^='additional_" + add + "']").selectpicker('refresh');
    });
}

function getConfiguration() {
    var config = {
        session:defaults.session,
        dbms: readCheckbox("input[name='dbms']:checked"),
        wkld: $("input[name='workload']:checked").val(),
        spe: readCheckbox("input[name^='specific']:checked"),
        met: readCheckbox("input[name='metric']:checked"),
        knb: readCheckbox("input[name='knob']:checked"),
        nres: $("#results_per_page option:selected").val(),
        eq: $("#equidistant").is(':checked') ? "on" : "off"
    };
    config["add"] = [];
    $.each(defaults.additional, function(i, add) {
        config["add"].push(add + ":" + $("select[name^='additional_" + add + "']").val());
    });

    return config;
}

function updateSub(event) {
    $("[id^=div_specific]").hide();
    $("input[name^='specific']").removeAttr('checked');
    var workload = $("input[name='workload']:checked").val();
    if (workload != "grid" && workload != "show_none") {
        $("div[id='div_specific_" + workload + "']").show();
        $("input[id^='specific_" + workload + "_']").prop('checked', true);
    }
}

function initializeSite(event) {
    setValuesOfInputFields(event);
    var mt = $("#metrictable").dataTable({
        "aaSorting": [],
        "bFilter": false,
        "bAutoWidth": true,
        "bDestroy": true,
        "iDisplayLength": 35,
        "aLengthMenu": [[35, 50, 100, -1], [35, 50, 100, "All"]]
    });
    $("#results_per_page"                ).bind('change', updateUrl);
    $("input[name='dbms']"          ).bind('click', updateUrl);
    $("input[name='workload']"   ).on('change', updateSub);
    $("input[name='workload']"   ).on('click', updateUrl);
    $("input[name^='specific']"   ).on('change', updateUrl);
    $("select[name^='additional']").bind('change', updateUrl);
    $("input[name='metric']"   ).on('click', updateUrl);
    $("input[name='knob']"   ).on('click', updateUrl);
    $("#equidistant"              ).bind('change', updateUrl);
    
}

function refreshSite(event) {
    setValuesOfInputFields(event);
    refreshContent();

}

function setValuesOfInputFields(event) {
    // Either set the default value, or the one parsed from the url

    // Set default selected recent results
    $("#results_per_page").val(valueOrDefault(event.parameters.nres, defaults.results_per_page));
    $('#results_per_page').selectpicker('refresh');

    // Set default selected metrics
    $("input:checkbox[name='metric']").prop('checked', false);
    var metrics = event.parameters.met ? event.parameters.met.split(',') : defaults.metrics;
    $("input:checkbox[name='metric']").each(function() {
        if ($.inArray($(this).val(), metrics) >= 0) {
            $(this).prop('checked', true);
        }
    });
    
    // Set default selected dbms
    $("input:checkbox[name='dbms']").removeAttr('checked');
    var dbmss = defaults.dbms.split(',');
    var sel = $("input[name='dbms']");
    $.each(dbmss, function(i, db) {
        sel.filter("[value='" + db + "']").prop('checked', true);
    });

    // Set default selected workload
    var workload = valueOrDefault(event.parameters.wkld, defaults.workload);
    $("input:radio[name='workload']").filter("[value='" + workload + "']").attr('checked', true);
    $("[id^=div_specific]").hide();
    $("input[name^='specific']").removeAttr('checked');
    if ($("input[name='workload']:checked").val() != "grid" && $("input[name='workload']:checked").val() != "show_none") {
        $("[id=div_specific_" + $("input[name='workload']:checked").val() + "]").show();
        sel = $("[id^=specific_" + $("input[name='workload']:checked").val() + "_]");
        var specs = event.parameters.spe && event.parameters.spe != "none" ? event.parameters.spe.split(','): defaults.workloads.split(',');
        $.each(specs, function(i, spec) {
            sel.filter("[value='" + spec + "']").prop('checked', true);
        });
    }

    // Set default selected additional filter
    if (event.parameters.add) {
        var filters = event.parameters.add.split(',');
        $.each(filters, function(i, filter) {
            var kv = filter.split(':');
            var name = kv[0];
            var value = kv[1];
            $("select[name^='additional_" + name + "']").val(value);
            $("select[name^='additional_" + name + "']").selectpicker('refresh');
        });
    } else {
        $.each(defaults.additional, function(i, add) {
            $("select[name^='additional_" + add + "']").val("select_all");
            $("select[name^='additional_" + add + "']").selectpicker('refresh');
        });
    }

    // Set equidistant status
    $("#equidistant").prop('checked', valueOrDefault(event.parameters.eq, defaults.equidistant) === "on");
}

function init(def) {
    defaults = def;

    $.ajaxSetup ({
      cache: false
    });

    // Event listener for clicks on plot markers
    $.jqplot.eventListenerHooks.push(['jqplotClick', OnMarkerClickHandler]);

    // Init and change handlers are set to the refreshContent handler
    $.address.init(initializeSite).change(refreshSite);
    
}

return {
    init: init
};

})(window);
