
var selectedGraph = "";
var selectedGraphLocation = "";

function saveModel() {
  
  var val = document.getElementById("model-name").value;

  $.ajax({
    type: "POST",  
    url: "http://localhost:8000/model",
    data: {"graphname":val},
    success: function(response){  
      console.log(response);
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
  var node = {"key":nodename, "group":selectedGraph, "size": "60 60", "loc":selectedGraphLocation};

  var form  = $("#configForm").serialize();
  var params = form+"&graphname="+selectedGraph;

  if(diagram.model.containsNodeData(node)) {
     alert("Node already exists!");
  }
  else {
    $.ajax({
      type: "POST",  
      url: "http://localhost:8000/node",
      data: params,
      success: function(response){
        diagram.model.addNodeData(node);
        window.location.href = "#close";
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
      console.log(response);
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
      console.log(response);
    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }
  });

}

function removeNode(key, group) {

  $.ajax({
    type: "DELETE",  
    url:"http://localhost:8000/node",
    data: {
      "nodename": key, 
      "graphname": group
    },
    success: function(response){  
      console.log(response);
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
      console.log(response);
    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }
  });
}

function init (data) {
  
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
        console.log("node selected!");
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

        if (selectedGraph !== "Master") {
          window.location.href = "#openModal";
        }
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

  diagram.model = new go.GraphLinksModel(data.nodes, data.connections);
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
      removeNode(part.data.key, part.data.group);
    }

  });

  diagram.addDiagramListener("LinkDrawn",
  function(e) {

    var part = e.subject;
    addLink(part.data.from, part.data.to);

  });
}

function onSelectionChange(nodeCls) {

  $.ajax({
    //beforeSend: function() { textreplace(description); },
    type: "GET",  
    url: "http://localhost:8000/attr",
    data: {"nodeclass":nodeCls},
    success: function(response){  
     document.getElementById('node-config-div').innerHTML = response;
    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
    }       
  });

  // var wo_div = document.getElementById("weight-once-div");
  // var lf_div = document.getElementById("lf-nodes-div");
  // var dn_div = document.getElementById("data-node-div");

  // if(cls=="WeightNode" || cls=="OnesNode") {
  //   lf_div.style.display = "none";
  //   dn_div.style.display = "none";
  //   wo_div.style.display = "block";
  // }
  // else if(cls=="LossNode" || cls=="FunctionNode") {
  //   wo_div.style.display = "none";
  //   dn_div.style.display = "none";
  //   lf_div.style.display = "block";
  // }
  // else if(cls=="DataNode") {
  //   wo_div.style.display = "none";
  //   lf_div.style.display = "none";
  //   dn_div.style.display = "block";
  // }
  // else {
  //   wo_div.style.display = "none";
  //   lf_div.style.display = "none";
  //   dn_div.style.display = "none";
  // }

}