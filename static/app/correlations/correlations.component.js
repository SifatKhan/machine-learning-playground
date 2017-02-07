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
                    var div = Plotly.d3.select("div[id='heatmap']").node();
                    if (div != null)  Plotly.Plots.resize(div);
                });

            });

        }]
});