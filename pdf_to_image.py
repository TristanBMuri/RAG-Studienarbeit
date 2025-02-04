import base64
import fitz

doc = fitz.open(path)
for i, page in enumerate(doc):
    pix = page.get_pixmap()
    pix.save(f'page_{i}.png')

encoded_images = []

for i in range(16):
    image_path = f'page_{i}.png'
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")
        encoded_images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })
messages = [
    {"role": "system", "content": "Extract the full text from the following document. Please put it in markdonw structure"},
    {"role": "user", "content": encoded_images}
]

# response =