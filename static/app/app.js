var app = angular.module('myApp', ['ngRoute', 'ngFileUpload']);

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
        }).when('/classifier', {
            template: '<image-classifier></image-classifier>'
        }).otherwise('/neighborhoods');
    }
]);


