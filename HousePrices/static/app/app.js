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
        when('/pricingData', {
          template: '<pricing-data></pricing-data>'
        }).
        when('/correlations', {
          template: '<correlations></correlations>'
        }).
        otherwise('/neighborhoods');
    }
  ]);

angular.
  module('myApp').
  component('neighborhoodList', {
    templateUrl: 'static/app/neighborhood-list/neighborhood-list.template.html',
    controller: ['$scope','$http','$window',
    function NeighborhoodListController($scope,$http,$window) {

      $http.get('/api/neighborhood/counts').then(function(response) {
        layout = {
            yaxis: {title: 'Count'},
            title: 'Total Houses Sold by Neighborhood'
        };
        Plotly.plot('graph',[response.data],layout)
      });

      $http.get('/api/neighborhood/boxplot').then(function(response) {
        layout = {
            yaxis: {title: '$USD'},
            title: 'Sale Prices by Neighborhood'
        };
        Plotly.plot('boxplot',response.data,layout)

        angular.element($window).bind('resize',function(){
            var div = Plotly.d3.select("div[id='graph']").node();
            Plotly.Plots.resize( div)

            div = Plotly.d3.select("div[id='boxplot']").node();
            Plotly.Plots.resize( div)

        });
      });
    }]
  });


  angular.
  module('myApp').
  component('yearlyData', {
    templateUrl: 'static/app/yearly-data/yearly-data.template.html',
    controller: ['$scope','$http','$window',
    function YearlyDataController($scope,$http,$window) {

      $http.get('/api/yearlyAvg').then(function(response) {
        Plotly.plot('yearlyMean',[response.data])

        angular.element($window).bind('resize',function(){
            var div = Plotly.d3.select("div[id='yearlyMean']").node();
            Plotly.Plots.resize( div)

        });
      });
    }]
  });


  angular.
  module('myApp').
  component('pricingData', {
    templateUrl: 'static/app/pricing-data/pricing-data.template.html',
    controller: ['$scope','$http','$window',
    function PricingDataController($scope,$http,$window) {


      $http.get('/api/pricing/lotfrontage').then(function(response) {
        layout = {
            yaxis: {title: '$USD'},
            xaxis: {title: 'Area in feet²'},
            title: 'Sales Price by Lot Frontage'
        };
        Plotly.plot('lotFrontage',response.data,layout);
      });

      $http.get('/api/pricing/lotarea').then(function(response) {
        layout = {
            yaxis: {title: '$USD'},
            xaxis: {title: 'Area in feet²'},
            title: 'Sales Price by Lot Area'
        };
        Plotly.plot('lotArea',response.data,layout);

        angular.element($window).bind('resize',function(){
            var div = Plotly.d3.select("div[id='lotArea']").node();
            Plotly.Plots.resize( div)

            div = Plotly.d3.select("div[id='lotFrontage']").node();
            Plotly.Plots.resize( div)
        });
      });

    }]
  });

  angular.
  module('myApp').
  component('correlations', {
    template: '<div class="row"> <div id="heatmap"></div> </div>',
    controller: ['$scope','$http','$window',
    function CorrelationsController($scope,$http,$window) {



      $http.get('/api/correlation').then(function(response) {
        layout = {
            title: 'Correlation Matrix',
            height: 1000,
            margin: {
            l: 100,
          },
            autosize: true,
        };
        Plotly.plot('heatmap',response.data,layout);

        angular.element($window).bind('resize',function(){
            var heat = Plotly.d3.select("div[id='heatmap']").node();
            Plotly.Plots.resize( heat)
        });


      });


    }]
  });
