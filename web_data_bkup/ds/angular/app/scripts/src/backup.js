var newNodes = [];
var removedNodes = [];
var newLinks = [];
var removedLinks = [];
var changedKeys = [];

function addNewNode(data) {

 return $.ajax({
    //beforeSend: function() { textreplace(description); },
    type: "POST",  
    url: "http://localhost:8000/awok/data",
    success: function(response){  
      console.log(response);
      alert("success");
      create_tree(response);
    },
    error: function(XMLHttpRequest, textStatus, errorThrown) { 
      console.log(XMLHttpRequest);
      alert("Error: " + errorThrown); 
    }       
  });

}

function addNode() {

  console.log(newNodes);
  console.log(removedNodes);
  console.log(newLinks);
  console.log(removedLinks);
  console.log(changedKeys);
  //diagram.model.addNodeData({"key":key, "color":"white"});
  // diagram.startTransaction("make new link");
  // diagram.model.addLinkData({ from: "Alpha", to: "Beta" });
  // diagram.commitTransaction("make new link");
}

function nodeConfig() {
  window.location.href = "#openModal";
}

function addNodeToGraph(graph,points) {
  console.log(points);
  var node = {"key":"empty", "group":graph, "size": "10 10", "loc":points};
  //var link = {"from":graph, "to":"newNode"};

  if(diagram.model.containsNodeData(node)) {

    while(true) {

      var rand = Math.random().toString(36).substring(7);
      node["key"] = rand;
      //link["to"] = rand;

      if(!diagram.model.containsNodeData(node))
        break;
    }
  }

  diagram.model.addNodeData(node);
  //diagram.model.addLinkData(link);

}

function createGraph() {
  
  var val = document.getElementById("graph-name").value;

  var node = {"key":val, "isGroup":true, "loc":"-70 10"};
  var link = {"from":"Master", "to":val};
  
  if(diagram.model.containsNodeData(node)) {
    alert("Key already exists!");
  }
  else{

    diagram.model.addNodeData(node);
    diagram.model.addLinkData(link);

  }

}
function onSelectionChanged(node) {
  console.log(node);
    var icon = node.findObject("Icon");
    if (icon !== null) {
      if (node.isSelected)
        icon.fill = "cyan";
      else
        icon.fill = "lightgray";
    }
  }

function create_tree (data) {
  // body...
  var $ = go.GraphObject.make;
  diagram =
    $(go.Diagram, "myDiagramDiv",  // create a Diagram for the DIV HTML element
      {
        // position the graph in the middle of the diagram
        initialContentAlignment: go.Spot.TopCenter,
        // allow double-click in background to create a new node
        //"clickCreatingTool.archetypeNodeData": { key: "Node", color: "white" },
        // allow Ctrl-G to call groupSelection()
        //"commandHandler.archetypeGroupData": { text: "Group", isGroup: true, color: "blue" },
        // enable undo & redo
        //"undoManager.isEnabled": true
      });


   diagram.nodeTemplate =
    $(go.Node, "Auto",{
        selectionObjectName:"PH",
        //selectionChanged: onSelectionChanged,
        doubleClick: function(inputEvent, graphObject) {
          console.log("node selected!");
        },
      },
      new go.Binding("location", "loc", go.Point.parse),
      $(go.Shape, "Ellipse", 
        { 
            fill: "white", // the default fill, if there is no data-binding
            portId: "", 
            cursor: "pointer",  // the Shape is the port, not the whole Node
            // allow all kinds of links from and to this port
            fromLinkable: true, fromLinkableSelfNode: true, fromLinkableDuplicates: true,
            toLinkable: true, toLinkableSelfNode: true, toLinkableDuplicates: true,
        },
        new go.Binding("fill", "color")
      ),
      $(go.TextBlock,
        new go.Binding("text", "key")
      )
    );

  // diagram.groupTemplate =
  //   $(go.Group, "Vertical",
  //     { 
  //       isShadowed:false,
  //       ungroupable:false,
  //       defaultSeparatorPadding:5,
  //       // fromLinkable:true,
  //       // toLinkable:true,
  //       doubleClick: function(inputEvent, graphObject) {
          
  //         var diagram = graphObject.diagram;
  //         var graph = graphObject.sh.key;

  //         var x = graphObject.location.x+30;
  //         var y = graphObject.location.y+30;
  //         var pts = x.toString()+" "+y.toString();

  //         if (graph !== "Master") {
  //           addNodeToGraph(graph,pts);
  //           //nodeConfig();
  //         }
  //       }
  //     },
  //     $(go.Panel, "Auto",
  //       $(go.Shape, "RoundedRectangle",  // surrounds the Placeholder
  //         { 
  //           //parameter1: 14,
  //           fill: "rgba(128,128,128,0.33)" 
  //         },
  //         new go.Binding("desiredSize", "size", go.Size.parse)
  //       ),
  //       $(go.Placeholder,    // represents the area of all member parts,
  //         { 
  //           padding: 5
  //         },
  //         new go.Binding("location", "loc", go.Point.parse)
  //       )  // with some extra padding around them
  //     ),
  //     $(go.TextBlock,         // group title
  //       { 
  //         alignment: go.Spot.TopCenter,
  //         font: "Bold 12pt Sans-Serif" 
  //       },
  //       new go.Binding("text", "key")
  //     ),
  //     new go.Binding("location", "loc", go.Point.parse)
  //   );
  
  
   diagram.groupTemplate =
    $(go.Group, "Vertical",
      { selectionObjectName: "PH",
        locationObjectName: "PH",
        resizable: true,
        resizeObjectName: "PH",
        isShadowed:false,
        ungroupable:false,
        defaultSeparatorPadding:5,
        doubleClick: function(inputEvent, graphObject) {
          
          var diagram = graphObject.diagram;
          var graph = graphObject.sh.key;
          var x = graphObject.location.x+30;
          var y = graphObject.location.y+30;

          // document.getElementById("myDiagramDiv").onclick = function(e){
          //   x = e.pageX - this.offsetLeft;
          //   y = e.pageY - this.offsetTop;
          //   console.log(this.offsetLeft);
          //   console.log(this.offsetTop);
          // };

          var pts = x.toString()+" "+y.toString();

          if (graph !== "Master") {
            addNodeToGraph(graph,pts);
            //nodeConfig();
          }
        }
        //fromLinkable:true,
        //cursor: "pointer",
      },
      new go.Binding("location", "loc", go.Point.parse),
      $(go.TextBlock,  // group title
        { font: "Bold 12pt Sans-Serif" },
        new go.Binding("text", "key")),
      $(go.Shape,  // using a Shape instead of a Placeholder
        { name: "PH",
          fill: "lightyellow" },
        new go.Binding("desiredSize", "size", go.Size.parse).makeTwoWay(go.Size.stringify))
    );

  
  // the Model holds only the essential information describing the diagram
  diagram.model = new go.GraphLinksModel(data.nodes, data.connections);
  diagram.findNodeForKey("Master").deletable = false;

  diagram.model.addChangedListener(function(evt) {
    // ignore unimportant Transaction events
    console.log(evt.changes)
    if (!evt.isTransactionFinished) return;
    var txn = evt.object;  // a Transaction
    if (txn === null) return;
    // iterate over all of the actual ChangedEvents of the Transaction
    txn.changes.each(function(e) {      // record text edit
      console.log(e)
      // ignore any kind of change other than adding/removing a node
      if (e.modelChange == "nodeDataArray") {
        // record node insertions and removals      
        if (e.change === go.ChangedEvent.Insert) {
          console.log(evt.propertyName + " added node with key: " + e.newValue.key);
          newNodes.push({"key":e.newValue.key});
        } else if (e.change === go.ChangedEvent.Remove) {
          console.log(evt.propertyName + " removed node with key: " + e.oldValue.key);
          removedNodes.push({"key":e.oldValue.key});
        }
      } 
      else if(e.modelChange == "linkDataArray") {
        if (e.change === go.ChangedEvent.Insert) {
          console.log(evt.propertyName + " link added from: " + e.newValue.from + " to: " + e.newValue.to);
          newLinks.push({"from":e.newValue.from, "to":e.newValue.to});
        } else if (e.change === go.ChangedEvent.Remove) {
          console.log(evt.propertyName + " link removed from: " + e.oldValue.from + " to: " + e.oldValue.to);
          removedLinks.push({"from":e.oldValue.from, "to":e.oldValue.to});
        }
      }
      else if (evt.isTransactionFinished && evt.dr == "TextEditing" && e.Yl == "text") {
        console.log(evt.propertyName + "Key: " + e.oldValue + " replaced with: " + e.newValue);
        changedKeys.push({"oldkey":e.oldValue, "newkey":e.newValue});
      } 
      else {
        return
      }
      
    });
  });


// function create_master_graph(data) {

//   var $ = go.GraphObject.make;
//   diagram =
//     $(go.Diagram, "myDiagramDiv",  // create a Diagram for the DIV HTML element
//       {
//         // position the graph in the middle of the diagram
//         initialContentAlignment: go.Spot.TopCenter,
//         // allow double-click in background to create a new node
//         //"clickCreatingTool.archetypeNodeData": { key: "Node", color: "white" },
//         // allow Ctrl-G to call groupSelection()
//         //"commandHandler.archetypeGroupData": { text: "Group", isGroup: true, color: "blue" },
//         // enable undo & redo
//         "undoManager.isEnabled": true
//       });


//    diagram.groupTemplate =
//     $(go.Group, "Vertical",
//       { selectionObjectName: "PH",
//         locationObjectName: "PH",
//         resizable: true,
//         resizeObjectName: "PH",
//         isShadowed:false,
//         ungroupable:false,
//         defaultSeparatorPadding:5,
//         doubleClick: function(inputEvent, graphObject) {
          
//           var diagram = graphObject.diagram;
//           var graph = graphObject.sh.key;
//           var x = graphObject.location.x+30;
//           var y = graphObject.location.y+30;

//           // document.getElementById("myDiagramDiv").onclick = function(e){
//           //   x = e.pageX - this.offsetLeft;
//           //   y = e.pageY - this.offsetTop;
//           //   console.log(this.offsetLeft);
//           //   console.log(this.offsetTop);
//           // };

//           var pts = x.toString()+" "+y.toString();

//           if (graph !== "Master") {
//             addNodeToGraph(graph,pts);
//             //nodeConfig();
//           }
//         }
//         //fromLinkable:true,
//         //cursor: "pointer",
//       },
//       new go.Binding("location", "loc", go.Point.parse),
//       $(go.TextBlock,  // group title
//         { font: "Bold 12pt Sans-Serif" },
//         new go.Binding("text", "key")),
//       $(go.Shape,  // using a Shape instead of a Placeholder
//         { name: "PH",
//           fill: "lightyellow" },
//         new go.Binding("desiredSize", "size", go.Size.parse).makeTwoWay(go.Size.stringify))
//     );

//   //diagram.layout = $(go.TreeLayout);
//   // the Model holds only the essential information describing the diagram
//   diagram.model = new go.GraphLinksModel(data.nodes, data.connections);
// }



  // the node template describes how each Node should be constructed
  // diagram.nodeTemplate =
  //   $(go.Node, "Auto",
  //     {
  //       selectable:true,
  //       selectionObjectName:"PH",
  //       selectionChanged: onSelectionChanged,
  //       doubleClick: function(inputEvent, graphObject) {
  //         console.log("node selected!");
  //       },
  //     },
  //     new go.Binding("location", "loc", go.Point.parse),
  //    //$(go.Panel, "Auto", //go.Node, "Auto",  // the Shape automatically fits around the TextBlock
  //       $(go.Shape, "Ellipse", //"RoundedRectangle",
  //         {
  //           fill: "white", // the default fill, if there is no data-binding
  //           portId: "", cursor: "pointer",  // the Shape is the port, not the whole Node
  //           // allow all kinds of links from and to this port
  //           fromLinkable: true, fromLinkableSelfNode: true, fromLinkableDuplicates: true,
  //           toLinkable: true, toLinkableSelfNode: true, toLinkableDuplicates: true,
  //         },
  //         // bind Shape.fill to Node.data.color
  //         new go.Binding("fill", "color")),
  //         //new go.Binding("minSize", "size", go.Size.parse)),
  //       $(go.TextBlock,
  //         { margin: 3, editable:false, font: "bold 11px sans-serif", stroke: '#333', margin: 6, isMultiline: false },  // some room around the text
  //         // bind TextBlock.text to Node.data.key
  //         new go.Binding("text", "key"))
  //    // ),
  //    // $("TreeExpanderButton")
  //    );
  
   // diagram.groupTemplate =
   //  $(go.Group, "Vertical",
   //    { 
   //      // selectionObjectName: "PH",
   //      // locationObjectName: "PH",
   //      // resizeObjectName: "PH",
   //      resizable: true,
   //      isShadowed:false,
   //      groupable:true,
   //      alignment: go.Spot.Top,
   //      //defaultSeparatorPadding:5,
   //      doubleClick: function(inputEvent, graphObject) {
          
   //        var diagram = graphObject.diagram;
   //        var graph = graphObject.sh.key;
   //        var x = graphObject.location.x+30;
   //        var y = graphObject.location.y+30;

   //        // document.getElementById("myDiagramDiv").onclick = function(e){
   //        //   x = e.pageX - this.offsetLeft;
   //        //   y = e.pageY - this.offsetTop;
   //        //   console.log(this.offsetLeft);
   //        //   console.log(this.offsetTop);
   //        // };

   //        var pts = x.toString()+" "+y.toString();

   //        if (graph !== "Master") {
   //          addNodeToGraph(graph,pts);
   //          //nodeConfig();
   //        }
   //      }
   //      //fromLinkable:true,
   //      //cursor: "pointer",
   //    },
   //    //new go.Binding("location", "loc", go.Point.parse),
   //    $(go.TextBlock,  // group title
   //      { font: "Bold 11pt Sans-Serif" },
   //      new go.Binding("text", "key")),
   //    $(go.Shape,  // using a Shape instead of a Placeholder
   //      { 
   //        name: "PH",
   //        fill: "lightyellow",
   //      }
   //      //new go.Binding("desiredSize", "size", go.Size.parse).makeTwoWay(go.Size.stringify)
   //      )
   //  );

  //diagram.layout = $(go.TreeLayout);

  // shiftNode = (function() {  // define a function named "shiftNode" callable by onclick
  //   // all model changes should happen in a transaction
  //   //diagram.startTransaction("shift node");
  //   var data = diagram.model.nodeDataArray[0];  // get the first node data
  //   var node = diagram.findNodeForData(data);   // find the corresponding Node
  //   console.log(node.location);
  //   // show the updated location held by the "loc" property of the node data
  //   //document.getElementById("bindTwoWayData").textContent = data.loc.toString();
  //   //diagram.commitTransaction("shift node");
  // });
  // shiftNode();  // initialize everything

 
}

function getClassAttributes(cls) {
  
}

function submitForm() {

  var form = $('#treeform');

  form.submit(function (ev) {

    var graphname = $('#graph-name').val();
    var nodename = $('#node-name').val();
    var nodeclass = $('#node-class').val();

    $.ajax({
      //beforeSend: function() { textreplace(description); },
      type: "POST",  
      url: "http://localhost:8000/awok/tree",
      data: {'graphname':graphname, 'nodename':nodename, 'nodeclass':nodeclass},
      success: function(response){  
        
      },
      error: function(XMLHttpRequest, textStatus, errorThrown) { 
       // create_tree({"nodes":[{"key":graphname, "isGroup":true}]});

        console.log(XMLHttpRequest);
      }       
    });
  });
}