angular.module('myApp').component('pricingData', {
    templateUrl: 'app/pricing-data/pricing-data.template.html',
    controller: ['$scope', '$http', '$window','$location',
        function PricingDataController($scope, $http, $window,$location) {

            $window.ga('send', 'pageview', {page: $location.url()});

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

            });

            window.onresize = function () {
              Plotly.Plots.resize(document.getElementById("lotArea"));
              Plotly.Plots.resize(document.getElementById("lotFrontage"));
              Plotly.Plots.resize(document.getElementById("grlivArea"));
              Plotly.Plots.resize(document.getElementById("histogram"));
          };

        }]
});