function onchange_fileSelector(event) {
  readURL(event);
}

function sampleImage_render(e) {
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

api_url = "http://127.0.0.1:5000/api/colorify";

async function submit() {
  img = document.getElementById("input-image");

  data = {
    gray_img: img.src,
  };

  console.log(data);

  let response = await fetch(api_url, {
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

console.log(imgs.img1);
