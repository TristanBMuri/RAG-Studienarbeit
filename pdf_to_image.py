import google.generativeai as genai
from PIL import Image

# Configure your API key
genai.configure(api_key='YOUR_API_KEY')  # Replace with your actual API key


# Function to process image
def analyze_image(image_path, prompt):
    # Load the image
    img = Image.open(image_path)

    # Initialize the model (use 'gemini-1.5-flash' or latest model name)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Send both image and text prompt
    response = model.generate_content([prompt, img])

    return response.text


# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Can be .jpg, .png, etc.
    prompt = "Describe this image in detail"

    try:
        result = analyze_image(image_path, prompt)
        print("Gemini Response:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")