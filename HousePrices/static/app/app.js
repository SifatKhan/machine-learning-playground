var app = angular.module('myApp', ['ngRoute']);

angular.module('myApp').config(['$locationProvider', '$routeProvider',
    function config($locationProvider, $routeProvider) {
        $locationProvider.hashPrefix('!');

        $routeProvider.when('/neighborhoods', {
            template: '<neighborhood-list></neighborhood-list>'
        }).when('/trends', {
            template: '<trends></trends>'
        }).when('/pricingData', {
            template: '<pricing-data></pricing-data>'
        }).when('/correlations', {
            template: '<correlations></correlations>'
        }).otherwise('/neighborhoods');
    }
]);


angular.module('myApp').component('trends', {
    templateUrl: 'app/trends/trends.template.html',
    controller: ['$scope', '$http', '$window',
        function TrendsController($scope, $http, $window) {

            $http.get('/api/yearly/count').then(function (response) {

                layout = {
                    yaxis: {title: 'Count'},
                    xaxis: {title: 'Year Sold'},
                    title: 'Houses Sold per Year'
                };
                Plotly.plot('yearlyCount', [response.data], layout, {displayModeBar: false})

            });

            $http.get('/api/monthly/count').then(function (response) {

                layout = {
                    yaxis: {title: 'Count'},
                    xaxis: {title: 'Month Sold', tickformat: "%B", dtick: "M1"},
                    title: 'Houses Sold by Month'
                };
                Plotly.plot('monthlyCount', [response.data], layout, {displayModeBar: false})

            });

            $http.get('/api/yearly/avg').then(function (response) {
                layout = {
                    yaxis: {title: '$USD'},
                    xaxis: {title: 'Year Sold'},
                    title: 'Average Sale Price per Year'
                };
                Plotly.plot('yearlyMean', [response.data], layout, {displayModeBar: false})

                angular.element($window).unbind('resize');
                angular.element($window).bind('resize', function () {
                    var div = Plotly.d3.select("div[id='yearlyMean']").node();
                    Plotly.Plots.resize(div)

                    div = Plotly.d3.select("div[id='yearlyCount']").node();
                    Plotly.Plots.resize(div)

                    div = Plotly.d3.select("div[id='monthlyCount']").node();
                    Plotly.Plots.resize(div)

                });


            });
        }]
});


angular.module('myApp').component('pricingData', {
    templateUrl: 'app/pricing-data/pricing-data.template.html',
    controller: ['$scope', '$http', '$window',
        function PricingDataController($scope, $http, $window) {


            $http.get('/api/pricing/lotfrontage').then(function (response) {
                layout = {
                    yaxis: {title: '$USD'},
                    xaxis: {title: 'Area in feet²'},
                    title: 'Sales Price by Lot Frontage'
                };
                Plotly.plot('lotFrontage', response.data, layout, {displayModeBar: false});
            });


            $http.get('/api/pricing/histogram').then(function (response) {
                var mean = response.data.mean;
                var data = response.data.data;

                layout = {
                    yaxis: {title: '$USD'},
                    title: 'Sale Prices',
                    annotations: [
                        {
                            text: 'Mean: ' + mean,
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


                Plotly.plot('histogram', data, layout, {displayModeBar: false});
            });

            $http.get('/api/pricing/grlivarea').then(function (response) {
                layout = {

                    yaxis: {title: '$USD'},
                    xaxis: {title: 'Area in feet²'},
                    title: 'Above ground living area',
                    autosize: true
                };
                Plotly.plot('grlivArea', response.data, layout, {displayModeBar: false});
            });

            $http.get('/api/pricing/lotarea').then(function (response) {
                layout = {

                    yaxis: {title: '$USD'},
                    xaxis: {title: 'Area in feet²'},
                    title: 'Sales Price by Lot Area'
                };
                Plotly.plot('lotArea', response.data, layout, {displayModeBar: false});

                angular.element($window).unbind('resize');
                angular.element($window).bind('resize', function () {
                    var div = Plotly.d3.select("div[id='lotArea']").node();
                    Plotly.Plots.resize(div);

                    div = Plotly.d3.select("div[id='lotFrontage']").node();
                    Plotly.Plots.resize(div);

                    div = Plotly.d3.select("div[id='grlivArea']").node();
                    Plotly.Plots.resize(div);

                    div = Plotly.d3.select("div[id='histogram']").node();
                    Plotly.Plots.resize(div);
                });
            });

        }]
});

angular.module('myApp').component('correlations', {
    template: '<div class="row"> <div id="heatmap"></div> </div>',
    controller: ['$scope', '$http', '$window',
        function CorrelationsController($scope, $http, $window) {


            $http.get('/api/correlation').then(function (response) {
                layout = {
                    title: 'Correlation Matrix',
                    height: 1000,
                    margin: {
                        l: 100,
                    },
                    autosize: true,
                };
                Plotly.plot('heatmap', response.data, layout, {displayModeBar: false});

                angular.element($window).unbind('resize');
                angular.element($window).bind('resize', function () {
                    var heat = Plotly.d3.select("div[id='heatmap']").node();
                    Plotly.Plots.resize(heat);
                });

            });

        }]
});
