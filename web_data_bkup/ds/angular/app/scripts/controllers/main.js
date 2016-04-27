'use strict';

/**
 * @ngdoc function
 * @name angularApp.controller:MainCtrl
 * @description
 * # MainCtrl
 * Controller of the angularApp
 */
angular.module('angularApp')
  .controller('MainCtrl', ['$scope', '$http', function ($scope, $http) {
  	var data = {'nodes': [
	    { key: 1, name: "Alpha", color: "lightblue" },
	    { key: 2, name: "Beta", color: "orange" },
	    { key: 3, name: "Gamma", color: "lightgreen" },
	    { key: 4, name: "Delta", color: "pink" }
	  ],
	  'connections':[
	    { from: 1, to: 2 },
	    { from: 1, to: 3 },
	    { from: 2, to: 2 },
	    { from: 3, to: 4 },
	    { from: 4, to: 1 }
	  ]}

  	init(data);

  	$scope.nodeClasses = ['AttentionNode', 'ConvolutionNode', 'DataNode', 
  		'DotProductNode', 'FilterNode', 'FunctionNode', 'HadamardNode', 
  		'LossNode', 'OnesNode', 'PoolNode', 'TargetNode', 'VectorAddNode', 
  		'VectorsToMatrixNode', 'WeightNode']

  	$scope.methods = {
  		createGraph : function() {
  			alert($scope.graphname);
  		},
  		addNode : function() {
  			alert('addNode');
  		},
  		addLink : function() {
  			alert("addLink");
  		},
  		deleteGraph : function() {
  			alert("deleteGraph");
  		},
  		deleteNode : function() {
  			alert('deleteNode');
  		},
  		deleteLink : function() {
  			alert('deleteLink');
  		}
  	};

    function init (data) {

		  var $ = go.GraphObject.make;

		  $scope.diagram = $(go.Diagram, "myDiagramDiv",
		    {
		      // position the graph in the middle of the diagram
		      initialContentAlignment: go.Spot.TopLeft,
		      "undoManager.isEnabled": true,
		    }
		  );

		  $scope.diagram.nodeTemplate = $(go.Node, "Auto",{
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

		  $scope.diagram.linkTemplate = $(go.Link,
		    { routing: go.Link.Scale, corner: 10 },
		    $(go.Shape, { strokeWidth: 2 }),
		    $(go.Shape, { toArrow: "OpenTriangle" })
		  );

		  $scope.diagram.groupTemplate = $(go.Group, "Auto",
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
			
		  $scope.diagram.model = new go.GraphLinksModel(data.nodes, data.connections);
		  $scope.diagram.layout = $(go.TreeLayout);
		  $scope.diagram.layout.arrangement = go.TreeLayout.ArrangementHorizontal;

		  $scope.diagram.addDiagramListener("SelectionDeleting",
		  function(e) {

		    if (e.diagram.selection.count > 1) {
		      e.cancel = true;
		      console.log("Cannot delete multiple selected parts");
		      return;
		    }

		    var part = e.diagram.selection.iterator.first();

		    if (part instanceof go.Link) {
		      $scope.methods.deleteLink(part.data.from, part.data.to);
		    }
		    else if (part instanceof go.Group) {
		      $scope.methods.deleteGraph(part.data.key);
		    }
		    else if (part instanceof go.Node){
		      $scope.methods.deleteNode(part.data.key, part.data.group);
		    }

		  });

		  $scope.diagram.addDiagramListener("LinkDrawn",
		  function(e) {

		    var part = e.subject;
		    addLink(part.data.from, part.data.to);

		  });
		}

		$scope.showModal = false;
    $scope.toggleModal = function(){
        $scope.showModal = !$scope.showModal;
    };

}])
.directive('nodeConfiguration', [
  function () {

    return {
      restrict: 'EA',
      templateUrl: '../../views/directives/config.html',
      replace: true,
      scope: {
      }
    };

  }
]);
// .directive('modal', function () {
//     return {
//       template: '<div class="modal fade">' + 
//           '<div class="modal-dialog">' + 
//             '<div class="modal-content">' + 
//               '<div class="modal-header">' + 
//                 '<button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button>' + 
//                 '<h4 class="modal-title">{{ title }}</h4>' + 
//               '</div>' + 
//               '<div class="modal-body" ng-transclude></div>' + 
//             '</div>' + 
//           '</div>' + 
//         '</div>',
//       restrict: 'E',
//       transclude: true,
//       replace:true,
//       scope:true,
//       link: function postLink(scope, element, attrs) {
//         scope.title = attrs.title;

//         scope.$watch(attrs.visible, function(value){
//           if(value == true)
//             $(element).modal('show');
//           else
//             $(element).modal('hide');
//         });

//         $(element).on('shown.bs.modal', function(){
//           scope.$apply(function(){
//             scope.$parent[attrs.visible] = true;
//           });
//         });

//         $(element).on('hidden.bs.modal', function(){
//           scope.$apply(function(){
//             scope.$parent[attrs.visible] = false;
//           });
//         });
//       }
//     };
//   });
