var app = angular.module('myApp', ['ngRoute']);

angular.
  module('myApp').
  config(['$locationProvider', '$routeProvider',
    function config($locationProvider, $routeProvider) {
      $locationProvider.hashPrefix('!');

      $routeProvider.
        when('/neighborhoods', {
          template: '<neighborhood-list></neighborhood-list>'
        }).
        when('/yearlyAvg', {
          template: '<yearly-data></yearly-data>'
        }).
        otherwise('/neighborhoods');
    }
  ]);

angular.
  module('myApp').
  component('neighborhoodList', {
    templateUrl: 'static/app/neighborhood-list/neighborhood-list.template.html',
    controller: ['$scope','$http',
    function NeighborhoodListController($scope,$http) {

      $http.get('/api/neighborhood/counts').then(function(response) {
        $scope.neighborhoods = response.data;
        Plotly.plot('graph',[$scope.neighborhoods])
      });

      $http.get('/api/neighborhood/boxplot').then(function(response) {
        $scope.neighborhoods = response.data;
        Plotly.plot('boxplot',$scope.neighborhoods)
      });
    }]
  });


  angular.
  module('myApp').
  component('yearlyData', {
    templateUrl: 'static/app/yearly-data/yearly-data.template.html',
    controller: ['$scope','$http',
    function NeighborhoodListController($scope,$http) {

      $http.get('/api/yearlyAvg').then(function(response) {
        $scope.neighborhoods = response.data;
        Plotly.plot('yearlyMean',[$scope.neighborhoods])
      });
    }]
  });
