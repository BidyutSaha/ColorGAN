from __future__ import print_function
import requests
import json
import matplotlib.pyplot as plt
import base64

addr = 'http://localhost:5000'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
img = plt.imread('testImg.png')
# encode image as jpeg
img_b64 = base64.b64encode(img)
print(type(img_b64.tostring()))
# send http request with image and receive response
response = requests.post(
    test_url, data=img_b64, headers=headers)
# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}
