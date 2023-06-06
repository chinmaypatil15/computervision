import os
import requests
from bs4 import BeautifulSoup
import urllib

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def scrape_images(url, category):
    # Send a GET request to the URL
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the image elements on the page
    images = soup.find_all('img')

    # Download the images
    for i, img in enumerate(images):
        img_url = img['src']
        filename = f"{category}_{i}.jpg"
        download_image(img_url, filename)

def download_image(url, filename):
    # Send a GET request to download the image
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the image to a file
        with open(filename, 'wb') as file:
            for chunk in response:
                file.write(chunk)
        print(f"Image downloaded: {filename}")
    else:
        print(f"Failed to download image: {url}")


# Define the URLs and categories
urls = ['https://www.flipkart.com/apple-iphone-14-midnight-128-gb/p/itm9e6293c322a84?pid=MOBGHWFHECFVMDCX&lid=LSTMOBGHWFHECFVMDCX8AI8PX&marketplace=FLIPKART&q=iphones&store=tyy%2F4io&srno=s_1_1&otracker=search&otracker1=search&iid=0397af2c-4b3a-41cc-a1af-4efb956e0d45.MOBGHWFHECFVMDCX.SEARCH&ssid=6nqlh3x6z40000001685444169715&qH=3e7fa8c51e2e4986', 'https://www.flipkart.com/apple-iphone-13-midnight-128-gb/p/itmca361aab1c5b0?pid=MOBG6VF5Q82T3XRS&lid=LSTMOBG6VF5Q82T3XRSOXJLM9&marketplace=FLIPKART&q=iphones&store=tyy%2F4io&spotlightTagId=BestsellerId_tyy%2F4io&srno=s_1_2&otracker=search&otracker1=search&fm=Search&iid=0397af2c-4b3a-41cc-a1af-4efb956e0d45.MOBG6VF5Q82T3XRS.SEARCH&ppt=sp&ppn=sp&ssid=6nqlh3x6z40000001685444169715&qH=3e7fa8c51e2e4986']
categories = ['category1', 'category2']

# Create directories to store the images
for category in categories:
    os.makedirs(category, exist_ok=True)

# Scraping images from each URL
for url, category in zip(urls, categories):
    scrape_images(url, category)


# Define the parameters for the data generator
batch_size = 32
image_size = (128, 128)

# Create an image data generator
data_generator = ImageDataGenerator(rescale=1./255)

# Load the dataset from the directories
train_dataset = data_generator.flow_from_directory(
    directory='./dataset/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Create a custom model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)


