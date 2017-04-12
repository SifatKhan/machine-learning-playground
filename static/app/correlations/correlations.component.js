angular.module('myApp').component('correlations', {
    template: '<div> <div id="heatmap"></div> </div>',
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

            });

            window.onresize = function () {
              Plotly.Plots.resize(document.getElementById("heatmap"));
          };

        }]
});