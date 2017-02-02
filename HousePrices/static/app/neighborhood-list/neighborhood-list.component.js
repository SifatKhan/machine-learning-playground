angular.module('myApp').component('neighborhoodList', {
    templateUrl: 'app/neighborhood-list/neighborhood-list.template.html',
    controller: ['$scope', '$http', '$window',
        function NeighborhoodListController($scope, $http, $window) {

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

                angular.element($window).unbind('resize');
                angular.element($window).bind('resize', function () {
                    var div = Plotly.d3.select("div[id='graph']").node();
                    if (div != null) Plotly.Plots.resize(div);

                    div = Plotly.d3.select("div[id='boxplot']").node();
                    if (div != null) Plotly.Plots.resize(div);

                });
            });
        }]
});