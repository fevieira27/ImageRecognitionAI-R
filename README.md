# ImageTagAI-R
Script to load a local image, use TensorFlow/Keras to identify people, pets and objects, creating tags based on the ImageNet classification database (from a total of 1000 tags available).

There are now 3 types of scripts:
- A **simple** one that uses only a single AI model;
- A more **complex** that uses 15 models at the same time, knowing that depending on the image some models work better than others. This script aims to get the classification tags from those more prevalent and with higher prediction confidence accross all models;
![Example of complex script](./img/Example.jpg)
- The third script has all 15 models above, but is also using **Google and Azure Vision AI** (API keys required) to further improve tags and to identify landmarks and addresses.

![Example of complex script with Vision APIs for Location](./img/Location_Example.jpg)

Future update ideas:
- Based on all the above knowledge about the image, generate a social media text using AI with location and hashtags for easy posting.
