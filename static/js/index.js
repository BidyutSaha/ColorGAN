var isSample = false;
var sampleId = 1;
var api_url = "/api/colorify";
var api_url_sample = "/api/sampleColorify";

function onchange_fileSelector(event) {
  isSample = false;
  readURL(event);
}

function sampleImage_render(e) {
  isSample = true;
  sampleId = e.getAttribute("tag");
  img1 = document.getElementById("input-image");
  img1.src = e.src;

  img = document.getElementById("img-col-1");
  $("html, body").animate(
    {
      scrollTop: $("#mid").offset().top,
    },
    2000
  );
}

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      img = document.getElementById("input-image");
      img.src = input_image_content_b64 = e.target.result;
      $("html, body").animate(
        {
          scrollTop: $("#mid").offset().top,
        },
        2000
      );
    };

    reader.readAsDataURL(input.files[0]);
  }
}

function op_render() {
  img1 = document.getElementById("input-image");
  img = document.getElementById("img-col-1");
  img.src = img1.src;
}

async function submit() {
  $("html, body").animate(
    {
      scrollTop: $("#footer").offset().top,
    },
    2000
  );
  var ele_loading = document.getElementById("loading");
  var ele_result = document.getElementById("result");

  ele_loading.hidden = false;
  ele_result.hidden = true;
  console.log(ele_loading, ele_result);
  url = "";
  data = null;
  if (isSample) {
    data = {
      gray_img_id: sampleId,
    };
    url = api_url_sample;
  } else {
    img = document.getElementById("input-image");
    data = {
      gray_img: img.src,
    };
    url = api_url;
  }

  let response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json;charset=utf-8",
    },
    body: JSON.stringify(data),
  });

  let result = await response.json();

  img = document.getElementById("img-col-2");
  img.src = result.img1;

  img = document.getElementById("img-col-1");
  img.src = result.img2;
  ele_loading.hidden = true;
  ele_result.hidden = false;
}

/**
 * This handler retrieves the images from the clipboard as a base64 string and returns it in a callback.
 *
 * @param pasteEvent
 * @param callback
 */
function retrieveImageFromClipboardAsBase64(pasteEvent, callback, imageFormat) {
  if (pasteEvent.clipboardData == false) {
    if (typeof callback == "function") {
      callback(undefined);
    }
  }

  var items = pasteEvent.clipboardData.items;

  if (items == undefined) {
    if (typeof callback == "function") {
      callback(undefined);
    }
  }

  for (var i = 0; i < items.length; i++) {
    // Skip content if not image
    if (items[i].type.indexOf("image") == -1) continue;
    // Retrieve image on clipboard as blob
    var blob = items[i].getAsFile();

    // Create an abstract canvas and get context
    var mycanvas = document.createElement("canvas");
    var ctx = mycanvas.getContext("2d");

    // Create an image
    var img = new Image();

    // Once the image loads, render the img on the canvas
    img.onload = function () {
      // Update dimensions of the canvas with the dimensions of the image
      mycanvas.width = this.width;
      mycanvas.height = this.height;

      // Draw the image
      ctx.drawImage(img, 0, 0);

      // Execute callback with the base64 URI of the image
      if (typeof callback == "function") {
        callback(mycanvas.toDataURL(imageFormat || "image/png"));
      }
    };

    // Crossbrowser support for URL
    var URLObj = window.URL || window.webkitURL;

    // Creates a DOMString containing a URL representing the object given in the parameter
    // namely the original Blob
    img.src = URLObj.createObjectURL(blob);
  }
}

window.addEventListener("load", function () {
  window.addEventListener(
    "paste",
    function (e) {
      // Handle the event
      retrieveImageFromClipboardAsBase64(e, function (imageDataBase64) {
        // If there's an image, open it in the browser as a new window :)
        if (imageDataBase64) {
          // data:image/png;base64,iVBORw0KGgoAAAAN......
          isSample = false;
          img = document.getElementById("input-image");
          img.src = imageDataBase64;
          $("html, body").animate(
            {
              scrollTop: $("#mid").offset().top,
            },
            2000
          );
        }
      });
    },
    false
  );
});
