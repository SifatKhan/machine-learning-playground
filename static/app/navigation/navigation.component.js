angular.module('myApp').component('navigation', {
    templateUrl: 'app/navigation/navigation.template.html',
    controller: ['$scope','$location',
        function NavigationController($scope,$location) {
            $scope.currentPath = $location.path();
        }]
});

