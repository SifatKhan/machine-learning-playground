var app = angular.module('myApp', ['ui.router', 'ui.bootstrap', 'ngRoute', 'ngFileUpload']);

angular.module('myApp').config(['$locationProvider', '$routeProvider', '$stateProvider','$urlRouterProvider',
    function config($locationProvider, $routeProvider, $stateProvider, $urlRouterProvider) {

        $stateProvider.state('housing', {
            url: "/housing",
            templateUrl: 'app/navigation/housing-nav.template.html'
        }).state('housing.neighborhoods',{
            url: "/neighborhoods",
            template: '<neighborhood-list></neighborhood-list>'
        }).state('housing.trends',{
            url: "/trends",
            template: '<trends/>'
        }).state('housing.pricingData',{
            url: "/pricingData",
            template: '<pricing-data/>'
        }).state('housing.correlations',{
            url: "/correlations",
            template: '<correlations/>'
        }).state('classifier', {
            url: "/classifier",
            templateUrl: 'app/navigation/classifier-nav.template.html'
        }).state('classifier.main',{
            url: "/main",
            template: '<image-classifier/>'
        }).state('classifier.about', {
            url: "/about",
            templateUrl: 'app/about/aboutclassifier.template.html'
        });

        $urlRouterProvider.when('', '/housing/neighborhoods');
    }
]);


