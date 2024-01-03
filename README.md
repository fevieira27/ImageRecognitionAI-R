# ImageAI-R
R scripts developed to load a local image, identify people/pets/objects and create tags based on the ImageNet classification database (from a total of 1000 tags available). Additionally, it tries to identify the landmark/location based on Google/Azure Vision and Google/Bing maps API, using all this information to automatically generate a prompt to Bing Chat.

There are 4 types of scripts:
- A **simple** one that uses only a single AI model for tag identification ([1-AI_image_tag_simple.R](1-AI_image_tag_simple.R));
  
- A second more **complex** that does tag identification using 15 models at the same time, based on the knowledge that, depending on what is contained in the image, some models work better than others. This script aims to get the classification tags from those more prevalent and with higher prediction confidence accross all models ([2-AI_image_tag_multimodal.R](2-AI_image_tag_multimodal.R));
![Example of complex script](./img/Example.jpg)

- The third script still uses all 15 models above, but is also using **Google and Azure Vision AI** (API keys required) to further improve tags and to identify landmarks and addresses ([3-AI_image_multimodal_location.R](3-AI_image_multimodal_location.R));
![Example of complex script with Vision APIs for Location](./img/Location_Example.jpg)

- Based on all the above knowledge about the image, the forth script improves the results further by also using **Google Maps API and Bing Maps API** (API keys required) to more accurately search for the location/landmark name based on GPS data and hashtags found on image. Finally, it generates a **prompt for Bing Chat** on the default browser requesting a social media text for easier posting, feeding the AI with probable location and hashtags identified ([4-AI_image_multimodal_location_GPT.R](4-AI_image_multimodal_location_GPT.R)).
![Example of complex script with Vision and Maps APIs for Location](./img/Location_GPT_Example.jpg)
![Example of GPT Prompt](./img/GPT_Example.jpg)
