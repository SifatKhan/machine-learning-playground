angular.module('myApp').component('neighborhoodList', {
    templateUrl: 'app/neighborhood-list/neighborhood-list.template.html',
    controller: ['$scope', '$http', '$window', '$location',
        function NeighborhoodListController($scope, $http, $window, $location) {

            $window.ga('send', 'pageview', {page: $location.url()});
            $http.get('/api/neighborhood/counts').then(function (response) {
                layout = {
                    yaxis: {title: 'Count'},
                    title: 'Total Houses Sold by Neighborhood'
                };
                Plotly.plot('graph', [response.data], layout, {displayModeBar: false})
            });

            $http.get('/api/neighborhood/boxplot').then(function (response) {
                layout = {
                    yaxis: {title: '$USD'},
                    title: 'Sale Prices by Neighborhood',
                    height: 800
                };
                Plotly.plot('boxplot', response.data, layout, {displayModeBar: false})
            });

            window.onresize = function () {
                Plotly.Plots.resize(document.getElementById("graph"));
                Plotly.Plots.resize(document.getElementById("boxplot"));
            };

        }]
});