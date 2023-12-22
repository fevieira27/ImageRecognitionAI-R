# ImageTagAI-R
Script to load a local image, use TensorFlow/Keras to identify people, pets and objects, creating tags based on the ImageNet classification database (from a total of 1000 tags available).

There are 2 types of scripts:
- A **simple** one that uses only a single AI model
- A more **complex** that uses 15 models at the same time, knowing that depending on the image some models work better than others. This script aims to get the classification tags from those more prevalent and with higher prediction confidence accross all models.


Future update ideas:
- Get location from latitude/longitude and use some kind of vision AI to identify buildings or landscapes;
- Based on all the above knowledge about the image, generate a social media text using AI with location and hashtags for easy posting.
