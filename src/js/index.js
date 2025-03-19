window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 10000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})


$(document).ready(function() {
// Check for click events on the navbar burger icon
$(".navbar-burger").click(function() {
	// Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
	$(".navbar-burger").toggleClass("is-active");
	$(".navbar-menu").toggleClass("is-active");

});

var options = {
		slidesToScroll: 1,
		slidesToShow: 3,
		loop: true,
		infinite: true,
		autoplay: false,
		autoplaySpeed: 3000,
}

	// Initialize all div with carousel class
var carousels = bulmaCarousel.attach('.carousel', options);

// Loop on each carousel initialized
for(var i = 0; i < carousels.length; i++) {
	// Add listener to  event
	carousels[i].on('before:show', state => {
		console.log(state);
	});
}

// Access to bulmaCarousel instance of an element
var element = document.querySelector('#my-element');
if (element && element.bulmaCarousel) {
	// bulmaCarousel instance is available as element.bulmaCarousel
	element.bulmaCarousel.on('before-show', function(state) {
		console.log(state);
	});
}

/*var player = document.getElementById('interpolation-video');
player.addEventListener('loadedmetadata', function() {
	$('#interpolation-slider').on('input', function(event) {
	console.log(this.value, player.duration);
	player.currentTime = player.duration / 100 * this.value;
	})
}, false);*/
preloadInterpolationImages();

$('#interpolation-slider').on('input', function(event) {
	setInterpolationImage(this.value);
});
setInterpolationImage(0);
$('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

bulmaSlider.attach();

})

// eyes retargeting

window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE_EYES = "./src/img/eyes_retargeting/";
var NUM_INTERP_FRAMES = 4800;


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    preloadInterpolationImageseyes();

    $('#interpolation-slider-eyes').on('input', function(event) {
      setInterpolationImageeyes(this.value);
    });
    setInterpolationImageeyes(0);
    $('#interpolation-slider-eyes').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})


var INTERP_BASE_LIP = "./src/img/geo_optimization/";

var interp_images_lip = [];
function preloadInterpolationImageslip() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i=i+10) {
    // var path = INTERP_BASE_LIP + '/' + String(i) + '.jpg';
    var filename = String(i).padStart(6, '0') + '.png'; // 转换为 06d 格式
    var path = INTERP_BASE_LIP + '/' + filename;
    interp_images_lip[i] = new Image();
    interp_images_lip[i].src = path;
  }
}

function setInterpolationImagelip(i) {
  var image_lip = interp_images_lip[i];
  image_lip.ondragstart = function() { return false; };
  image_lip.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-lip').empty().append(image_lip);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    // preloadInterpolationImageslip();

    $('#interpolation-slider-lip').on('input', function(event) {
      setInterpolationImagelip(this.value);
    });
    setInterpolationImagelip(0);
    $('#interpolation-slider-lip').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
