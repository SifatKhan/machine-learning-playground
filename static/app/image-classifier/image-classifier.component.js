angular.module('imageClassifier', ['ngFileUpload']);

angular.module('myApp').component('imageClassifier', {
    templateUrl: 'app/image-classifier/image-classifier.template.html',
    controller: ['$scope', 'Upload', '$timeout', function ($scope, Upload, $timeout) {
        $scope.uploadPic = function (file) {
            file.upload = Upload.upload({
                url: './api/imageupload',
                data: {file: file},
            });

            file.upload.then(function (response) {
                $timeout(function () {
                    file.result = response.data.results;
                });
            }, function (response) {
                if (response.status > 0)
                    $scope.errorMsg = response.status + ': ' + response.data;
            }, function (evt) {
                // Math.min is to fix IE which reports 200% sometimes
                file.progress = Math.min(100, parseInt(100.0 * evt.loaded / evt.total));
            });
        }
    }]
});