
var selectedGraph = "";
var selectedGraphLocation = "";

function createNewModel() {
  
  var val = document.getElementById("model-name").value;

  var exists = false;

  $('#model-selection option').each(function(){
      if (this.value == val) {
          exists = true;
          return;
      }
  });

  if (!exists) {

    window.location.href = "#close";

    $.ajax({
      type: "POST",  
      url: "http://localhost:8000/model",
      data: {"modelname":val},
      success: function(response){  
        window.location.href = "./home";
      },
      error: function(XMLHttpRequest, textStatus, errorThrown) { 
        console.log(XMLHttpRequest);
      }       
    });
  } else {
    alert('Name already exists!');
  }
}

function copyModel() {
  
  var val = document.getElementById("copy-model-name").value;

  var exists = false;
  $('#model-selection option').each(function(){
      if (this.value == val) {
          exists = true;
          return;
      }
  });

  if (!exists) {
   
    window.location.href = "#close";
    
    $.ajax({
      type: "PUT",  
      url: "http://localhost:8000/model",
      data: {"modelname":val},
      success: function(response){  
        window.location.href = "./home";
        // diagram.clear();
        // diagram.model = new go.GraphLinksModel(response.nodes, response.connections);
        // diagram.isReadOnly = !(response['allowEdit']);
      },
      error: function(XMLHttpRequest, textStatus, errorThrown) { 
        console.log(XMLHttpRequest);
      }       
    });
  } else {
    alert('Name already exists!');
  }
}

function updateNodeConfig() {

  var params  = $("#editConfigForm").serialize();
  window.location.href = "#close";
  
  $.ajax({
    type: "PUT",  
    url: "http://localhost:8000/node",
    data: params,
    success: function(response){
    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }
  });

}

function createGraph() {
  
  var val = document.getElementById("graph-name").value;

  var node = {"key":val, "isGroup":true, "size":"200 200", "loc":"120 -80"};
  
  if(diagram.model.containsNodeData(node)) {
    alert("Key already exists!");
  }
  else{

    $.ajax({
      type: "POST",  
      url: "http://localhost:8000/graph",
      data: {"graphname":val},
      success: function(response){
        diagram.model.addNodeData(node);
      },
      error: function(XMLHttpRequest, textStatus, errorThrown) { 
        console.log(XMLHttpRequest);
      }       
    });
    
  }
}

function addNodeToGraph() {

  var nodename = $('#node-name').val();
  var nodeclass = document.getElementById('selected-class').value;

  var node = {"key":nodename, "group":selectedGraph, "size": "60 60", "loc":selectedGraphLocation};

  var form  = $("#configForm").serialize();
  var params = form+"&graphname="+selectedGraph+"&nodeclass="+nodeclass;

  if(diagram.model.containsNodeData(node)) {
     alert("Node already exists!");
  }
  else {

    window.location.href = "#close";

    $.ajax({
      type: "POST",  
      url: "http://localhost:8000/node",
      data: params,
      success: function(response){
        diagram.model.addNodeData(node);
      },
      error: function(XMLHttpRequest, textStatus, errorThrown) { 
        console.log(XMLHttpRequest);
      }
    });
  }
}

function addLink(parentKey, childKey) {

  var parentNode = diagram.findNodeForKey(parentKey);
  var childNode = diagram.findNodeForKey(childKey);

  $.ajax({
    type: "POST",  
    url: "http://localhost:8000/link",
    data: {
      "parentnode": parentKey, "parentgraph":parentNode.data.group,
      "childnode":childKey, "childgraph":childNode.data.group
    },
    success: function(response){

    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }
  });
}

function removeGraph(key) {

  $.ajax({
    type: "DELETE",  
    url:"http://localhost:8000/graph",
    data: {
      "graphname": key, 
    },
    success: function(response){  

    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }
  });
}

function removeNode(key, group, parents) {

  parent_rel = JSON.stringify(parents)

  $.ajax({
    type: "DELETE",  
    url:"http://localhost:8000/node",
    data: {
      "nodename": key, 
      "graphname": group,
      "parents": parent_rel
    },
    success: function(response){  

    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }
  });

}

function removeLink(parentKey, childKey) {

  var parentNode = diagram.findNodeForKey(parentKey);
  var childNode = diagram.findNodeForKey(childKey);

  $.ajax({
    type: "DELETE",  
    url: "http://localhost:8000/link",
    data: {
      "parentnode": parentKey, "parentgraph":parentNode.data.group,
      "childnode":childKey, "childgraph":childNode.data.group
    },
    success: function(response){  

    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }
  });
}

function init () {

  var $ = go.GraphObject.make;

  diagram = $(go.Diagram, "myDiagramDiv",
    {
      // position the graph in the middle of the diagram
      initialContentAlignment: go.Spot.TopLeft,
      "undoManager.isEnabled": true,
    }
  );

  diagram.nodeTemplate = $(go.Node, "Auto",{
      doubleClick: function(inputEvent, graphObject) {
        data = graphObject.part.data;
        editNodeConfig(data.group, data.key);
      },
    },
    new go.Binding("location", "loc", go.Point.parse),
    $(go.Shape, "Ellipse", 
      { 
          fill: "white",
          portId: "",
          cursor: "pointer",
          fromLinkable: true, fromLinkableSelfNode: false, fromLinkableDuplicates: false,
          toLinkable: true, toLinkableSelfNode: false, toLinkableDuplicates: false,
      },
      new go.Binding("fill", "color"),
      new go.Binding("desiredSize", "size", go.Size.parse)
    ),
    $(go.TextBlock,
      new go.Binding("text", "key")
    )
  );

  diagram.linkTemplate = $(go.Link,
    { routing: go.Link.Scale, corner: 10 },
    $(go.Shape, { strokeWidth: 2 }),
    $(go.Shape, { toArrow: "OpenTriangle" })
  );

  diagram.groupTemplate = $(go.Group, "Auto",
    { 
      isShadowed:false,
      ungroupable:false,
      defaultSeparatorPadding:5,
      layout: $(go.TreeLayout,
      {
        angle: 90,
        arrangement: go.TreeLayout.ArrangementHorizontal,
        isRealtime: false
      }),
      isSubGraphExpanded: true,
      doubleClick: function(inputEvent, graphObject) {
        
        var diagram = graphObject.diagram;
        var x = graphObject.location.x+30;
        var y = graphObject.location.y+30;

        selectedGraph = graphObject.sh.key;
        selectedGraphLocation = x.toString()+" "+y.toString();
        window.location.href = "#node-config";
      }
    },
   $(go.Shape, "RoundedRectangle",
      { fill: "lightgrey", stroke: "gray", strokeWidth: 2 }),
   new go.Binding("minSize", "size", go.Size.parse),
    $(go.Panel, "Vertical",
      { defaultAlignment: go.Spot.Left, margin: 4 },
      $(go.Panel, "Horizontal",
        { defaultAlignment: go.Spot.Top },
        $("SubGraphExpanderButton"),
        $(go.TextBlock,
          { font: "Bold 18px Sans-Serif", margin: 4 },
          new go.Binding("text", "key"))
      ),
      // create a placeholder to represent the area where the contents of the group are
      $(go.Placeholder,
        { padding: new go.Margin(0, 10) })
    )  // end Vertical Panel
  );  // end Group

  
  diagram.layout = $(go.TreeLayout);
  diagram.layout.arrangement = go.TreeLayout.ArrangementHorizontal;

  diagram.addDiagramListener("SelectionDeleting",
  function(e) {

    if (e.diagram.selection.count > 1) {
      e.cancel = true;
      console.log("Cannot delete multiple selected parts");
      return;
    }

    var part = e.diagram.selection.iterator.first();

    if (part instanceof go.Link) {
      removeLink(part.data.from, part.data.to);
    }
    else if (part instanceof go.Group) {
      removeGraph(part.data.key);
    }
    else if (part instanceof go.Node){
      var parents = getParentNodes(part);
      removeNode(part.data.key, part.data.group, parents);
    }

  });

  diagram.addDiagramListener("LinkDrawn",
  function(e) {

    var part = e.subject;
    addLink(part.data.from, part.data.to);

  });
}

function getParentNodes(deleteNode)
{
    var json_object = {};

    var nodeGroup = deleteNode.data.group;
    var allParents = deleteNode.findLinksInto();

    while (allParents.next())
    {
        var parent = allParents.value;
        var parentNode = diagram.findNodeForKey(parent.data.from);
        var dict_key = parentNode.data.group;

        if (dict_key in json_object) {
          json_object[dict_key].push(parentNode.data.key);
        }
        else {
          json_object[dict_key] = [parentNode.data.key];
        }
    }

    return json_object;
}

function editNodeConfig(graph_name, node_name) {

  data = {"graphname":graph_name, "nodename":node_name};

  var model = document.getElementById('model-selection').value;

  if (model != "-- Select Model --") {
    data['modelname'] = model;
  }

  $.ajax({
    type: "GET",  
    url: "http://localhost:8000/node",
    data: data,
    success: function(response){  
      if(response['html'] == false) {
        document.getElementById('edit-config-div').innerHTML = "<p>No configuration for this node!</p>";
      } else {
        document.getElementById('edit-config-div').innerHTML = response;
      }
      window.location.href = "#edit-node-config";
    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }       
  });
}

function onSelectionChange(nodeCls) {

  if (nodeCls != "-- Select Model --") {
    show_loader();
    $.ajax({
      type: "GET",  
      url: "http://localhost:8000/attr",
      data: {"nodeclass":nodeCls},
      success: function(response){  
       document.getElementById('node-config-div').innerHTML = response;
       hide_loader();
      },
      error: function(XMLHttpRequest, textStatus, errorThrown) { 
        console.log(XMLHttpRequest);
      }       
    });
  }
  else{
    return;
  }
  
}

function onSelectionModel(modelName) {

  $.ajax({
    type: "GET",  
    url: "http://localhost:8000/model",
    data: {"modelname":modelName},
    success: function(response){
      diagram.clear();
      diagram.model = new go.GraphLinksModel(response.nodes, response.connections);
      //diagram.isReadOnly = !(response['allowEdit']);

      if (document.getElementById('hidden-copy-btn')) {
        document.getElementById('hidden-copy-btn').style.display = 'inline';
        document.getElementById('hidden-start-graph-btn').style.display = 'inline';
      }

      if (diagram.isReadOnly && document.getElementById('graph-sbmt-btn')) {
        document.getElementById('graph-name').style.display = 'none';
        document.getElementById('graph-sbmt-btn').style.display = 'none';
      }
      else {
        document.getElementById('graph-name').style.display = 'inline';
        document.getElementById('graph-sbmt-btn').style.display = 'inline';
      }

    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }       
  });
}

function startGraph() {

  // /diagram.isReadOnly = true;

  if (document.getElementById('hidden-start-graph-btn')) {
    document.getElementById('hidden-start-graph-btn').disabled=true;
  }
  else if (document.getElementById('start-graph-btn')) {
    document.getElementById('start-graph-btn').disabled=true;
  }

  $.ajax({
    type: "GET",  
    url: "http://localhost:8000/activity",
    success: function(response){  

    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }       
  });

}

function stopGraph() {

  //diagram.isReadOnly = false;

  $.ajax({
    type: "GET",  
    url: "http://localhost:8000/activity",
    data: {"stop":true},
    success: function(response){  
      
    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }       
  });

}

function show_loader(){
  var loading_spinner = document.getElementById("loading");
  loading_spinner.style.visibility = 'visible';
}

function hide_loader(){
  var loading_spinner = document.getElementById("loading");
  loading_spinner.style.visibility = 'hidden';
}