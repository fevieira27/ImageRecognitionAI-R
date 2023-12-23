# When running first time, please follow this tutorial first:
# https://tensorflow.rstudio.com/install/
#
# After that, run the below command in R:
# remotes::install_github("rstudio/tensorflow")
#
# More information about the prediction models used, their parameters and expected results can also be found here:
# https://keras.io/api/applications/
# https://tensorflow.rstudio.com/reference/keras/#applications
# 
# Models being used:
# - MobileNetV3
# - VGG16
# - VGG19
# - ResNet-50
# - ResNet-101
# - ResNet-152
# - ResNet50V2
# - ResNet101V2
# - ResNet152V2
# - DenseNet201
# - Xception
# - Inception-ResNet-v2
# - Inception-V3
# - NasNetLarge
# - EfficientNet B7
# - Azure Vision AI
# - Google Cloud Vision

# Cleans all variables
rm(list = ls())

# Load the necessary libraries
library(reticulate)
library(remotes)
library(tensorflow)
library(keras)
library(tfdatasets)
library(dplyr)
library(imager)
library(magick)
library(AzureVision)
library(googleAuthR)
library(googleCloudVisionR)
library(ggmap)
library(devtools)
library(pkgbuild)
library(exifr)
library(geosphere)

# Set the API keys, first row only required on the first ever run of the script
  # Sys.setenv(AZURE_COMPUTER_VISION_KEY = "XXXXXXXXXXX")
  vis <- computervision_endpoint(
    url="https://XXXXXXXXXXX.cognitiveservices.azure.com/",
    key="XXXXXXXXXXX"
  )
  register_google(key = "XXXXXXXXXXX")
  options(googleAuthR.client_id = "XXXXXXXXXXX.apps.googleusercontent.com")
  options(googleAuthR.client_secret = "XXXXXXXXXXX")
  options(googleAuthR.scopes.selected = c("https://www.googleapis.com/auth/cloud-platform"))

# Authenticate with the Google Cloud Vision API
if (gar_has_token()!=TRUE) {
  gar_auth()
}

############## RUN ONLY UNTIL HERE FIRST. IF ALL GOES WELL, RUN THE REMAINING BELOW ##############

# Load the image
image_path <- "C:/Users/XXXXXXXXXXX/XXXXXXXXXXX.jpg"

# Plots the image being analysed
while (!is.null(dev.list())) { dev.off() }
img <- load.image(image_path)
# Get the image dimensions
info <- image_info(magick::image_read(file.path(image_path)))
# Check if the image is vertical
if (info$height > info$width) {
  # Rotate the image by 90 degrees clockwise
  img_rotated <- imrotate(img, -90)
} else {
  # Keep the image as it is
  img_rotated <- img
}
plot(img)

# Resizing image based on model requirements
image_224 <- image_load(image_path, target_size = c(224, 224))
image_299 <- image_load(image_path, target_size = c(299, 299))
image_331 <- image_load(image_path, target_size = c(331, 331))
image_600 <- image_load(image_path, target_size = c(600, 600))

#-------------------------------------------------------- Models that use 224x224 images
image <- image_to_array(image_224)
image <- array_reshape(image, c(1, dim(image)))

############################### No Preprocessing Models
# Load the pre-trained models
#9 (best) - mobilenet_v3
model <- application_mobilenet_v3_large(input_shape = c(224, 224, 3), alpha = 1, minimalistic = FALSE, include_top = TRUE, weights = "imagenet", input_tensor = NULL, classes = 1000L, pooling = NULL, dropout_rate = 0.2, classifier_activation = "softmax", include_preprocessing = TRUE) 
# Predict the top 5 classes
preds <- model %>% predict(image)
#tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
tags <- mobilenet_v2_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "mobilenet_v3"
# Save on a dataframe with results from all models
concat <- filtered_tags
############################## END No Preprocessing 


############################## Default Preprocessing Models
# Preprocess the image
image <- imagenet_preprocess_input(image)

# Load the pre-trained models

#10 - vgg16
model <- application_vgg16(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, pooling = NULL, classes = 1000, classifier_activation = "softmax") 
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "vgg16"
# Append results
concat <- rbind(concat,filtered_tags)

#11 - vgg19
model <- application_vgg19(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, pooling = NULL, classes = 1000, classifier_activation = "softmax") 
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "vgg19"
# Append results
concat <- rbind(concat,filtered_tags)
############################## END Default Preprocessing


############################## ResNet Preprocessing Models
# Preprocess the image
image <- image_to_array(image_224)
image <- array_reshape(image, c(1, dim(image)))
image <- resnet_preprocess_input(image)

# Load the pre-trained models
#1 - resnet50
model <- application_resnet50(include_top=TRUE, weights="imagenet", input_shape=c(224, 224, 3), classes=1000)
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "resnet50"
# Append results
concat <- rbind(concat,filtered_tags)

#2 (good) - resnet101
model <- application_resnet101(include_top=TRUE, weights="imagenet", input_shape=c(224, 224, 3), classes=1000)
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "resnet101"
# Append results
concat <- rbind(concat,filtered_tags)

#3 - resnet152
model <- application_resnet152(include_top=TRUE, weights="imagenet", input_shape=c(224, 224, 3), classes=1000)
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "resnet152"
# Append results
concat <- rbind(concat,filtered_tags)
############################## END ResNet Preprocessing


############################## ResNet V2 Preprocessing Models
# Preprocess the image
image <- image_to_array(image_224)
image <- array_reshape(image, c(1, dim(image)))
image <- resnet_v2_preprocess_input(image)

# Load the pre-trained models
#4 - resnet50_v2
model <- application_resnet50_v2(include_top=TRUE, weights="imagenet", input_shape=c(224, 224, 3), classes=1000, classifier_activation = "softmax")
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "resnet50_v2"
# Append results
concat <- rbind(concat,filtered_tags)

#5 - resnet101_v2
model <- application_resnet101_v2(include_top=TRUE, weights="imagenet", input_shape=c(224, 224, 3), classes=1000, classifier_activation = "softmax")
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "resnet101_v2"
# Append results
concat <- rbind(concat,filtered_tags)

#6 (best) - resnet152_v2
model <- application_resnet152_v2(include_top=TRUE, weights="imagenet", input_shape=c(224, 224, 3), classes=1000, classifier_activation = "softmax")
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "resnet152_v2"
# Append results
concat <- rbind(concat,filtered_tags)

############################## END ResNet V2 Preprocessing


############################## DenseNet Preprocessing Models
# Preprocess the image
image <- image_to_array(image_224)
image <- array_reshape(image, c(1, dim(image)))
image <- densenet_preprocess_input(image, data_format = NULL) 

# Load the pre-trained models
#8 (good) - densenet201
model <- application_densenet201(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, pooling = NULL, classes = 1000) 
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "densenet201"
# Append results
concat <- rbind(concat,filtered_tags)
############################## END DenseNet Preprocessing

#-------------------------------------------------------- Models that use 299x299 images
image <- image_to_array(image_299)
image <- array_reshape(image, c(1, dim(image)))

############################## Xception Preprocessing Model
# Preprocess the image
image <- xception_preprocess_input(image)

# Load the pre-trained models
#12 (chosen one) - xception
model <- application_xception( include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, pooling = NULL, classes = 1000, classifier_activation = "softmax") 
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "xception"
# Append results
concat <- rbind(concat,filtered_tags)
############################## END Xception Preprocessing


############################## Inception-ResNet V2 Preprocessing Model
# Preprocess the image
image <- image_to_array(image_299)
image <- array_reshape(image, c(1, dim(image)))
image <- inception_resnet_v2_preprocess_input(image)

# Load the pre-trained models
#13 (new) - inception_resnet_v2
model <- application_inception_resnet_v2(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, pooling = NULL, classes = 1000, classifier_activation = "softmax") 
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "inception_resnet_v2"
# Append results
concat <- rbind(concat,filtered_tags)
############################## END Inception-ResNet V2 Preprocessing


############################## Inception V3 Preprocessing Model
# Preprocess the image
image <- image_to_array(image_299)
image <- array_reshape(image, c(1, dim(image)))
image <- inception_v3_preprocess_input(image)

# Load the pre-trained models
#15 (new) - inception_v3
model <- application_inception_v3(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, pooling = NULL, classes = 1000, classifier_activation = "softmax")
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "inception_v3"
# Append results
concat <- rbind(concat,filtered_tags)
############################## END Inception V3 Preprocessing


#-------------------------------------------------------- Models that use 331x331 images
image <- image_to_array(image_331)
image <- array_reshape(image, c(1, dim(image)))

############################## NasNet Preprocessing Model
# Preprocess the image
image <- nasnet_preprocess_input(image) 

# Load the pre-trained models
#14 (new) - nasnetlarge
model <- application_nasnetlarge(input_shape = NULL, include_top = TRUE, weights = "imagenet", input_tensor = NULL, pooling = NULL, classes = 1000) 
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "nasnetlarge"
# Append results
concat <- rbind(concat,filtered_tags)
############################## END NasNet Preprocessing


#-------------------------------------------------------- Models that use 600x600 images
image <- image_to_array(image_600)
image <- array_reshape(image, c(1, dim(image)))

############################### No Preprocessing Models
# Load the pre-trained models
#7 (good) - efficientnet_b7
model <- application_efficientnet_b7(include_top = TRUE, weights = "imagenet", input_tensor=NULL, input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax") 
# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
# Filter only tags with higher than 10% confidence score
filtered_tags <- tags[tags$score >= 0.1,]
filtered_tags$model <- "efficientnet_b7"
# Append results
concat <- rbind(concat,filtered_tags)
############################### END No Preprocessing


############################### Cloud ML Models (Require API keys)
# Get tags from Azure Vision
  resultsAzure <- analyze(vis, image_path, domain = "landmarks", feature_types = "tags")$tags
  resultsAzure <- resultsAzure[, c("name", "confidence")]
  resultsAzure <- data.frame(class_name = "", resultsAzure)
  colnames(resultsAzure) <- c("class_name", "class_description", "score")
  resultsAzure$model <- "AzureVision"

# Append results from Azure Vision
concat <- rbind(concat,resultsAzure)

# Get tags from Google Vision
  resultsGoogle <- gcv_get_image_annotations(imagePaths = image_path, feature="LABEL_DETECTION", maxNumResults = 20)
  resultsGoogle <- resultsGoogle[, c("mid","description", "score")]
  colnames(resultsGoogle) <- c("class_name", "class_description", "score")
  resultsGoogle$class_description <- tolower(resultsGoogle$class_description)
  resultsGoogle$model <- "GoogleVision"

# Append results from Google
concat <- rbind(concat,resultsGoogle)
############################### END Cloud ML Models

# Raw data from results of all models
# print(concat)

# Create a dataframe with summarized results accross all models
results <- concat %>%
  # Group by column X
  group_by(class_description) %>%
  # Summarise the count and sum of column Y
  summarise(count = n(), sum_score = sum(score)) %>%
  # Keep only the distinct values of column X
  distinct(class_description, .keep_all = TRUE) %>%
  # Order by sum of scores, and then count if sum is equal (unlikely)
  arrange (desc(sum_score),desc(count))

print(results)

# Extract the hashtags for most prevalent (identified by more than 1 model) and with higher confidence leves (sum of scores higher than 0.4, and average score higher than 14.2%)
# No academic studies to support these parameters, but those worked best on my testing with both personal pictures and those from Pascal VOC dataset. Feel free to change them.
hashtags <- results[results$count > 1,]
hashtags <- hashtags[hashtags$sum_score >= 0.4,]
hashtags <- hashtags[hashtags$sum_score/hashtags$count >= 0.142,]
hashtags <- as.data.frame(hashtags)
hashtags <- paste0("#", tolower(gsub(" ", "", hashtags[, 1])))
hashtag_list <- hashtags

# Print the hashtags only from best results
hashtags <- paste(hashtag_list, collapse = " ")
# print(hashtags)

  # Read the EXIF data from the image file
  # exif_data <- read_exif(image_path, tags = c("filename", "GPSPosition", "GPSLatitude", "GPSLongitude"))
  exif_data <- read_exif(image_path)

# Check if GPS coordinates exist in exif data
if(all(c("GPSLatitude", "GPSLongitude") %in% names(exif_data))){
  exif_data <- exif_data[, c("FileName", "Directory", "GPSLatitude", "GPSLongitude")]
  exif_data <- as.data.frame(exif_data)
} else {
  exif_data <- exif_data[, c("FileName", "Directory")]
  exif_data$GPSLatitude <- ""
  exif_data$GPSLongitude <- ""
  exif_data <- as.data.frame(exif_data)
}

# Detect possible landmarks using Google Cloud Vision API
  landmarks <- gcv_get_image_annotations(imagePaths = image_path, feature="LANDMARK_DETECTION", maxNumResults = 10)
if(all(c("latitude", "longitude") %in% names(landmarks))){
  landmarks <- landmarks[, c("description", "score", "latitude", "longitude")]
} else {
  landmarks$latitude <- ""
  landmarks$longitude <- ""
}

# Check if the landmark list is empty
  if (landmarks$latitude[1] == "") {
    # If yes, set the landmark name to "Unknown" and take GPS coordinates from image exif data
    name <- "Unknown"
    if (exif_data$GPSLatitude != "") {
      lat <- exif_data$GPSLatitude
      lon <- exif_data$GPSLongitude
    } else { #exif GPS coordinates also not available, then blank
	lat <- ""
	lon <- ""
  } 
}else {
   if (exif_data$GPSLatitude != "") {
    # Identify which of the probable landmarks is closer to the actual image exif GPS data
    dist <- distm(landmarks[, c("longitude", "latitude")], exif_data[, c("GPSLongitude", "GPSLatitude")])
    closest_landmark <- landmarks[which.min(dist), ]

    # Extract the name and coordinates of the closest landmark from Google Vision
    name <- closest_landmark$description
    lat <- closest_landmark$latitude
    lon <- closest_landmark$longitude
   } else { # If exif GPS data is not available, then use only Google's
	# Sort landmarks by confidence score in descending order
	landmarks <- landmarks[order(landmarks$score, decreasing = TRUE), ]
  # Take the highest confidence one
	landmarks <- head(landmarks, 1)
	name <- landmarks$description
    	lat <- landmarks$latitude
    	lon <- landmarks$longitude
   }
}

# Reverse geocode the coordinates using ggmap, to get the address
if (lon!="") {
  address <- revgeocode(c(lon, lat), output = "all")
  city <- address$results[[1]]$address_components[[4]]$long_name
  country <- address$results[[1]]$address_components[[6]]$long_name
  address <- address$results[[1]]$formatted_address
} else { # If no GPS information was found anywhere
  address <- "Unknown"
  city <- "Unknown"
  country <- "Unknown"
}

# Extract text found in the image using Azure Computer Vision API (OCR)
  text <- read_text(vis, image_path, detect_orientation = TRUE, language = "en")
  # text <- analyze(vis, image_path, domain = "landmarks", feature_types = "description")$description$captions$text

# Show results
  cat(" Hashtags:     ", hashtags, "\n", "GPS Coordin.: ", lat, ",", lon, "\n", "Landmark Name:", name, "\n", "Text Found:   ", paste(text, collapse = ", "), "\n", "Full address: ", address, "\n", "City:         ", city, "\n", "Country:      ", country, "\n")

