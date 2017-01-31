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
            title: 'Sale Prices by Neighborhood',
            height: 800
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

      $http.get('/api/yearly/count').then(function(response) {

              layout = {
            yaxis: {title: 'Count'},
            xaxis: {title: 'Year Sold'},
            title: 'Houses Sold per Year'
        };
        Plotly.plot('yearlyCount',[response.data],layout)

      });

      $http.get('/api/yearly/avg').then(function(response) {
      layout = {
            yaxis: {title: '$USD'},
            xaxis: {title: 'Year Sold'},
            title: 'Average Sale Price per Year'
        };
        Plotly.plot('yearlyMean',[response.data],layout)

        angular.element($window).bind('resize',function(){
            var div = Plotly.d3.select("div[id='yearlyMean']").node();
            Plotly.Plots.resize( div)

            div = Plotly.d3.select("div[id='yearlyCount']").node();
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


      $http.get('/api/pricing/histogram').then(function(response) {
        var mean = response.data.mean;
        var data = response.data.data;

        layout = {
            yaxis: {title: '$USD'},
            title: 'Sale Prices',
            annotations: [
            { text: 'Mean: '+mean,
              x: mean,
              y: 150
            }
            ],
            shapes: [
            {
              x0: mean,
              x1: mean,
              xref: 'x',
              y1: 150,
              y0: 0
            }
            ]
        };


        Plotly.plot('histogram',data,layout);
      });

      $http.get('/api/pricing/grlivarea').then(function(response) {
        layout = {
            yaxis: {title: '$USD'},
            xaxis: {title: 'Area in feet²'},
            title: 'Above ground living area'
        };
        Plotly.plot('grlivArea',response.data,layout);
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

            div = Plotly.d3.select("div[id='grlivArea']").node();
            Plotly.Plots.resize( div)

            div = Plotly.d3.select("div[id='histogram']").node();
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
