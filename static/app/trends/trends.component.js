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
                    if (div != null) Plotly.Plots.resize(div);

                    div = Plotly.d3.select("div[id='yearlyCount']").node();
                    if (div != null) Plotly.Plots.resize(div);

                    div = Plotly.d3.select("div[id='monthlyCount']").node();
                    if (div != null) Plotly.Plots.resize(div);

                });

            });
        }]
});