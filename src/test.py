import base64

with open("/home/tmt/Documents/face/collection/Cong Anh/53491942_1224175431066305_4033200567001022464_n.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    print(encoded_string)
