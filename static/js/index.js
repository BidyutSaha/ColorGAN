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
}

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      img = document.getElementById("input-image");
      img.src = input_image_content_b64 = e.target.result;
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
}
