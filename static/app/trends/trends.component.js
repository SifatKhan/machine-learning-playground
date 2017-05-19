angular.module('myApp').component('trends', {
    templateUrl: 'app/trends/trends.template.html',
    controller: ['$scope', '$http', '$window','$location',
        function TrendsController($scope, $http, $window,$location) {

            $window.ga('send', 'pageview', {page: $location.url()});

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

            });

            window.onresize = function () {
              Plotly.Plots.resize(document.getElementById("monthlyCount"));
              Plotly.Plots.resize(document.getElementById("yearlyMean"));
              Plotly.Plots.resize(document.getElementById("yearlyCount"));
          };


        }]
});