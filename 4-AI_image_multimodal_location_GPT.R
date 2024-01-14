# When running first time, please follow this tutorial first:
# https://tensorflow.rstudio.com/install/
#
# After that, run the below command in R:
# remotes::install_github("rstudio/tensorflow")
# remotes::install_github("Azure/AzureCognitive")
#
# More information about the prediction models used, their parameters and expected results can also be found here:
# https://keras.io/api/applications/
# https://tensorflow.rstudio.com/reference/keras/#applications
# 
# Image tag models being used:
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

# Load the image
image_path <- "C:/XXXXX/PXL_20211221_115502968.jpg"

# Install/Load the necessary libraries
if (!require("revgeo")) { install.packages("revgeo") }
if (!require("httr")) { install.packages("httr") }
if (!require("reticulate")) { install.packages("reticulate") }
if (!require("remotes")) { install.packages("remotes") }
if (!require("tensorflow")) { install.packages("tensorflow") }
if (!require("keras")) { install.packages("keras") }
if (!require("tfdatasets")) { install.packages("tfdatasets") }
if (!require("dplyr")) { install.packages("dplyr") }
if (!require("imager")) { install.packages("imager") }
if (!require("magick")) { install.packages("magick") }
if (!require("AzureVision")) { install.packages("AzureVision") }
if (!require("AzureCognitive")) { install.packages("AzureCognitive") }
if (!require("googleAuthR")) { install.packages("googleAuthR") }
if (!require("googleCloudVisionR")) { install.packages("googleCloudVisionR") }
if (!require("ggmap")) { install.packages("ggmap") }
if (!require("devtools")) { install.packages("devtools") }
if (!require("pkgbuild")) { install.packages("pkgbuild") }
if (!require("exifr")) { install.packages("exifr") }
if (!require("geosphere")) { install.packages("geosphere") }
if (!require("revgeo")) { install.packages("revgeo") }
if (!require("purrr")) { install.packages("purrr") }

# Set the API keys
  # Sys.setenv(AZURE_COMPUTER_VISION_KEY = azure_api_key)
  # Sys.setenv("GCV_AUTH_FILE" = "/fullpath/to/auth.json")

  # Google Cloud API information
  options(googleAuthR.client_id = "XXXXX.apps.googleusercontent.com")
  options(googleAuthR.client_secret = "XXXXX")
  options(googleAuthR.scopes.selected = c("https://www.googleapis.com/auth/cloud-platform"))
  # Google Maps API key
  google_api_key <- readLines("C:/XXXXX/Google_API_key.txt", warn=FALSE)
  register_google(key = google_api_key)
  # Bing Maps API key
  bing_api_key <- readLines("C:/XXXXX/Bing_API_key.txt", warn=FALSE)
  # Azure API key
  azure_api_key <- readLines("C:/XXXXX/Azure_API_key.txt", warn=FALSE)
  # Azure Computer Vision API information
  cognitiveservicesURL <-"https://XXXXX.cognitiveservices.azure.com/"
  vis <- computervision_endpoint(
    url=cognitiveservicesURL,
    key=azure_api_key
  )

# Authenticate with the Google Cloud Vision API  (ONLY NEEDED FOR THE FIRST TIME OR IF TOKEN IS LOST)
if (gar_has_token()!=TRUE) {
  # This code will run automatically if no Google auth token is found. However, the token might exist but has expired, so if this is the case, then please run the below line manually
  gar_auth(email = "XXXXX")
}

############## IF EMAIL FOR TOKEN HAS NOT YET BEEN SETUP ABOVE, RUN ONLY UNTIL HERE FIRST AND ONLY AFTER RUN ALL THE BELOW SCRIPT ##############

# Set functions
# recursive function to remove name from all levels of list
stripname <- function(x, name) {
    thisdepth <- depth(x)
    if (thisdepth == 0) {
        return(x)
    } else if (length(nameIndex <- which(names(x) == name))) {
     	  x <- x[-nameIndex]
    }
    return(lapply(x, stripname, name))
}

# function to find depth of a list element
# see http://stackoverflow.com/questions/13432863/determine-level-of-nesting-in-r
depth <- function(this, thisdepth=0){
    if (!is.list(this)) {
        return(thisdepth)
    } else{
        return(max(unlist(lapply(this,depth,thisdepth=thisdepth+1))))    
    }
}

# Define a function to check if a sublist contains any of the words in a list
has_any_word <- function (x, words) {
  # If x is a character vector, use grepl to check for any of the words
  if (is.character (x)) {
    # Use paste to collapse the words into a single pattern separated by |
    return (any (grepl (paste (words, collapse = "|"), x, ignore.case = TRUE)))
  }
  # If x is a list, apply the function recursively to each element of the list
  if (is.list (x)) {
    return (any (sapply (x, has_any_word, words)))
  }
  # Otherwise, return FALSE
  return (FALSE)
}

# Plots the image being analysed
while (!is.null(dev.list())) { dev.off() }
img <- load.image(image_path)

# Get the image dimensions
info <- image_info(magick::image_read(file.path(image_path)))
# Check if the image is horizontal
if (info$width > info$height) {
  # Rotate the image by 90 degrees clockwise
  img_rotated <- imrotate(img, 90)
} else {
  # Keep the image as it is
  img_rotated <- img
}
plot(img_rotated) ## All my photos for social media are on the vertical position, but change this to "img" if you don't want to see your photos always plotted that way

# read the local image file as raw bytes, reducing to 4Mb (if needed) which is Azure's max limit
if (file.size(image_path)>4150000){
  raw_vector <- readBin(image_path, "raw", 4150000)
} else {
  raw_vector <- readBin(image_path, "raw", file.info(image_path)$size)
}

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
# Filter only tags with higher than 18% confidence score
filtered_tags <- tags[tags$score >= 0.18,]
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
# Filter only tags with higher than 16% confidence score
filtered_tags <- tags[tags$score >= 0.16,]
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
# Filter only tags with higher than 16% confidence score
filtered_tags <- tags[tags$score >= 0.16,]
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
  if (file.size(image_path)>4150000){ # needed?
    resultsAzure <- analyze(vis, raw_vector, domain = "landmarks", feature_types = "tags")$tags
  } else {
    resultsAzure <- analyze(vis, image_path, domain = "landmarks", feature_types = "tags")$tags
  }
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

# Raw data from results of all 15 models + 2 APIs
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

# print(results)

# Extract the hashtags for most prevalent (identified by more than 1 model) and with higher confidence leves (sum of scores higher than 0.4, and average score higher than 14.2%)
# No academic studies to support these parameters, but those worked best on my testing with both personal pictures and those from Pascal VOC dataset. Feel free to change them.
hashtags <- results[results$count > 1,]
hashtags <- hashtags[hashtags$sum_score >= 0.4,]
hashtags <- hashtags[hashtags$sum_score/hashtags$count >= 0.142,]
hashtags <- as.data.frame(hashtags)
hashwords <- paste0(tolower(gsub(" ", "_", hashtags[, 1])))
hashwords <- paste(hashwords, collapse = " ")
hashtags <- paste0("#", tolower(gsub(" ", "", hashtags[, 1])))
hashtag_list <- hashtags

# Create hashtags only from best results
hashtags <- paste(hashtag_list, collapse = " ")
# print(hashtags)

# Read the EXIF data from the image file
# exif_data <- read_exif(image_path, tags = c("filename", "GPSPosition", "GPSLatitude", "GPSLongitude"))
exif_data <- read_exif(image_path)

# Check if GPS coordinates exist in exif data
if(all(c("GPSLatitude", "GPSLongitude") %in% names(exif_data))){
  exif_data <- exif_data[, c("FileName", "Directory", "CreateDate", "Make", "Model", "GPSLatitude", "GPSLongitude")]
} else {
  exif_data <- exif_data[, c("FileName", "Directory", "CreateDate", "Make", "Model")]
  exif_data$GPSLatitude <- ""
  exif_data$GPSLongitude <- ""
}

# Format exif_data
exif_data <- as.data.frame(exif_data)
exif_data$CreateDate <- as.Date(exif_data$CreateDate, format = "%Y:%m:%d %H:%M:%S")
date <- format(exif_data$CreateDate, format = "%d/%B/%Y")

# Detect possible landmarks using Google Cloud Vision API
  landmarks <- gcv_get_image_annotations(imagePaths = image_path, feature="LANDMARK_DETECTION", maxNumResults = 10)
  if(all(c("latitude", "longitude") %in% names(landmarks))){ # Google Vision found landmark
    landmarks <- landmarks[, c("description", "score", "latitude", "longitude")]
    landmarks$source <- "Google Cloud Vision"
  } 

# Standalone Azure Computer Vision endpoint (must provide subscription key and cognitive services name)
endp <- cognitive_endpoint(cognitiveservicesURL,
    service_type="ComputerVision", key=azure_api_key)

# call the cognitive endpoint to analyze the image for landmarks
landmarkAzure <- call_cognitive_endpoint(endp, operation = "analyze",
	body = raw_vector,
	encode = "raw",
	options = list(details = "landmarks"),
	http_verb="POST")

if (length(landmarkAzure$categories[[1]]$detail$landmarks)>0){
  landmarkAzure <- dplyr::bind_rows(lapply(landmarkAzure$categories, as.data.frame.list))
  landmarkAzure <- landmarkAzure %>% select(detail.landmarks.name, detail.landmarks.confidence)

  # Keep only the 3 columns that are of interest
  landmarkAzure <- landmarkAzure[, grepl("landmarks.name|confidence", colnames(landmarkAzure))]

  # Rename, reorder and add blank columns to keep the df consistent with other solutions
  colnames(landmarkAzure) <- c("description", "score")
  landmarkAzure$latitude <- ""
  landmarkAzure$longitude <- ""
  landmarkAzure$source <- "Azure Vision API"
  landmarkAzure <- landmarkAzure %>% select(description, latitude, longitude, source, score)
}

# Add Azure Vision landmarks to the previous Google Vision results, if any result was found
if(exists("landmarkAzure") && all(c("description", "score") %in% names(landmarkAzure))){ # Azure Vision found landmarks
  landmarks <- rbind(landmarks, landmarkAzure, fill=TRUE)  
}

# Add Google Maps Places API landmarks to the previous Google Vision results
place_types <- c("accounting","airport","amusement_park","aquarium","art_gallery","atm","bakery","bank","bar","beauty_salon","bicycle_store","book_store","bowling_alley","bus_station","cafe","campground","car_dealer","car_rental","car_repair","car_wash","casino","cemetery","church","city_hall","clothing_store","convenience_store","courthouse","dentist","department_store","doctor","drugstore","electrician","electronics_store","embassy","fire_station","florist","funeral_home","furniture_store","gas_station","gym","hair_care","hardware_store","hindu_temple","home_goods_store","hospital","insurance_agency","jewelry_store","laundry","lawyer","library","light_rail_station","liquor_store","local_government_office","locksmith","lodging","meal_delivery","meal_takeaway","mosque","movie_rental","movie_theater","moving_company","museum","night_club","painter","park","pet_store","pharmacy","physiotherapist","plumber","police","post_office","primary_school","real_estate_agency","restaurant","roofing_contractor","rv_park","school","secondary_school","shoe_store","shopping_mall","spa","stadium","storage","store","subway_station","supermarket","synagogue","taxi_stand","tourist_attraction","train_station","transit_station","travel_agency","university","veterinary_care","zoo","landmark","place_of_worship","town_square")

# Check if any word from the list is in the text
result <- sapply(place_types, grepl, hashwords, fixed = TRUE)

# Print which words from the Google Places API were found in the hashtags based on the image
found_words <- place_types[result]
tourism_tags <- c("castle","monastery","bridge","palace","statue","bell_cote","hotel","fountain")
if (length(tourism_tags[sapply(tourism_tags, grepl, hashwords, fixed = TRUE)])>0) {
  found_words <- c(found_words, "tourist_attraction", "travel")
}

url <- ""
url <- paste0("https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=",
              exif_data$GPSLatitude, ",", exif_data$GPSLongitude,
              "&types=point_of_interest&rankby=distance&key=",google_api_key)

  # Send a GET request to the URL and store the response
  response <- GET(url)

  # Parse the response content as JSON
  content <- content(response, as = "parsed")

  # Filter landmark types based on hashtags
  filtered_content <- content$results
  filtered_content <- stripname(filtered_content, "vicinity")
  filtered_content <- stripname(filtered_content, "user_ratings_total")
  filtered_content <- stripname(filtered_content, "scope")
  filtered_content <- stripname(filtered_content, "reference")
  filtered_content <- stripname(filtered_content, "rating")
  filtered_content <- stripname(filtered_content, "plus_code")
  filtered_content <- stripname(filtered_content, "photos")
  filtered_content <- stripname(filtered_content, "opening_hours")
  filtered_content <- stripname(filtered_content, "icon_mask_base_uri")
  filtered_content <- stripname(filtered_content, "icon_background_color")
  filtered_content <- stripname(filtered_content, "icon")
  filtered_content <- stripname(filtered_content, "business_status")
  filtered_content <- stripname(filtered_content, "viewport")
  filtered_content <- stripname(filtered_content, "price_level")
  filtered_content <- stripname(filtered_content, "permanently_closed")

  # Use sapply to apply the logical test to each sublist of filtered_content
  result <- sapply(filtered_content, has_any_word, words = found_words)

  # Get the final subset based on original vector
  result <- filtered_content[result]

  # Extract the list of landmarks from the content
  df_land <- data.frame()

  # Convert to dataframe
  df_land <- dplyr::bind_rows(lapply(result, as.data.frame.list))

  # Keep only the 3 columns that are of interest
  df_land <- df_land[, grepl("name|lat|lng", colnames(df_land))]

  # Rename and reorder columns to keep the df consistent with other solution
  colnames(df_land) <- c("latitude", "longitude", "description")
  # landmarks <- df_land

if(exists("df_land") && all(c("latitude", "longitude") %in% names(df_land))){ # Google Places found landmarks
  df_land$source <- "Google Maps Places"
  df_land <- df_land %>% select(description, latitude, longitude, source)
  landmarks <- rbind(landmarks, df_land, fill=TRUE)  

} else { # If Google Places API was not able to find any landmark, try Bing maps API for nature-related POI

	# Define the URL for the Bing Location Recognition service
	url <- ""
	url <- paste0("https://dev.virtualearth.net/REST/v1/LocationRecog/",
              exif_data$GPSLatitude, ",", exif_data$GPSLongitude,
              "?o=json&includeEntityTypes=naturalPOI,businessAndPOI&key=",bing_api_key)

	# Send a GET request to the URL and store the response
	response <- GET(url)

	# Parse the response content as JSON
	content <- content(response, as = "parsed")

	# Extract the list of landmarks from the content
	landmarks2 <- content$resourceSets[[1]]$resources[[1]]$naturalPOIAtLocation

  if(length(landmarks2)!=0){

	# AlternateNames is not needed and was causing issues later on the script
	landmarks2 <- stripname(landmarks2, "alternateNames")

	# Initialize an empty dataframe
	df_land2 <- data.frame()

	# Loop over the elements of the vector and bind them by rows
	for (i in 1:length(landmarks2)) {
	  df_land2 <- rbind(df_land2, landmarks2[[i]])
	}

	colnames(df_land2) <- c("description", "latitude", "longitude", "type")
	df_land2$source <- "Bing Location Recognition"
	df_land2 <- df_land2 %>% select(description, latitude, longitude, source)

	if(exists("df_land2") && all(c("latitude", "longitude") %in% names(df_land2))){ # Bing Location has found nature POI
	    landmarks <- rbind(landmarks, df_land2, fill=TRUE)  
	}
  }
  
  # Also bringing business POI, to check GPS coordinates and try to find a suitable one
  businessPOI <- sapply(content$resourceSets[[1]]$resources[[1]]$businessesAtLocation, has_any_word, words = c(found_words, unlist(strsplit(hashwords, " "))))
  businessPOI <- content$resourceSets[[1]]$resources[[1]]$businessesAtLocation[businessPOI]
  businessPOI <- dplyr::bind_rows(lapply(businessPOI, as.data.frame.list))

  # Check if businessPOI returned something
  if (length(businessPOI)!=0){
    df_land3 <- businessPOI %>% select(businessInfo.entityName, businessAddress.latitude, businessAddress.longitude)
    colnames(df_land3) <- c("description", "latitude", "longitude")
    if(exists("df_land3") && all(c("latitude", "longitude") %in% names(df_land3))){ # Bing Location has found Business POI
      df_land3$source <- "Bing Location Recognition"
      landmarks <- rbind(landmarks, df_land3, fill=TRUE)  
    }
  }
}

if (length(landmarks$score)!=0){ # Google or Azure Vision have found landmarks
  landmarks <- landmarks %>% select(description, latitude, longitude, source, score)
} else {
    if (length(landmarks$description)!=0){ # Google Places or Bing Location have found possible POI
	landmarks <- landmarks %>% select(description, latitude, longitude, source)
	# removing rows with NA on the description
	landmarks <- landmarks[complete.cases(landmarks[ , description]),]	    
    } else {
	# If no landmarks were found so far, bring all Google Maps results (without key words search) and select based on distance to GPS coordinates
	df_land <- dplyr::bind_rows(lapply(filtered_content, as.data.frame.list))

	# Keep only the 3 columns that are of interest
	df_land <- df_land[, grepl("name|lat|lng", colnames(df_land))]

	# Rename and reorder columns to keep the df consistent with other solution
	colnames(df_land) <- c("latitude", "longitude", "description")

	if(exists("df_land") && all(c("latitude", "longitude") %in% names(df_land))){ # Google Places found landmarks
	  df_land$source <- "Google Maps Places"
	  df_land <- df_land %>% select(description, latitude, longitude, source)
	  landmarks <- rbind(landmarks, df_land, fill=TRUE)  
	} else {  
  	  landmarks$latitude <- ""
	  landmarks$longitude <- ""
	  landmarks$source <- ""
	}
    }
}
	
# Check if the landmark variable is empty
if (length(landmarks$latitude)==1 && landmarks$latitude=="") {
    # Set the name to "Unknown" and GPS coordinates from image exif data
    name <- "Unknown"
    source <- ""
    if (exif_data$GPSLatitude != "") {
      lat <- exif_data$GPSLatitude
      lon <- exif_data$GPSLongitude
    } else { #exif GPS coordinates also not available
	lat <- ""
	lon <- ""
    } 
}else { # Identify which of the probable landmarks is closer to the actual image exif GPS data
   if (length(landmarks$score)!=0){ # Google Vision has found landmarks, so use the highest score
		# sort df by score in descending order
		landmarks <- landmarks[order(landmarks$score, decreasing = TRUE), ]
		landmarks <- head(landmarks, 1)
		name <- landmarks$description
	    	lat <- landmarks$latitude
	    	lon <- landmarks$longitude
	   	source <- landmarks$source
   } else { # Find the POI from Google Places or Bing Location that is closest to GPS coordinates
   	if (exif_data$GPSLatitude != "") { # If there are GPS coordinates on image exif data    # Identify which of the probable landmarks is closer to the actual image exif data
	    dist <- distm(landmarks[, c("longitude", "latitude")], exif_data[, c("GPSLongitude", "GPSLatitude")])
	    closest_landmark <- landmarks[which.min(dist), ]

	    # Extract the name and coordinates of the landmark from Google Vision
	    name <- closest_landmark$description
	    source <- closest_landmark$source
	    # lat <- closest_landmark$latitude
	    # lon <- closest_landmark$longitude
	    lat <- exif_data$GPSLatitude
	    lon <- exif_data$GPSLongitude
	} else { # If there is no GPS data on exif and no Google Vision landmarks, use the first one brought by Google Places API or Bing Location
		landmarks <- head(landmarks, 1)
		name <- landmarks$description
	    	lat <- landmarks$latitude
	    	lon <- landmarks$longitude
	    	source <- landmarks$source		
	}
   }
}

# Find the address by reverse geocoding the coordinates using ggmap
if (lon!="") {
  address <- revgeocode(c(lon, lat), output = "all")
  if (address$results[[1]]$address_components[[1]]$long_name != ""){
    city_check <- sapply(address$results[[1]]$address_components, has_any_word, words = "locality")
    if (length(address$results[[1]]$address_components[city_check])!=0){
      city <- address$results[[1]]$address_components[city_check]
      city <- city[[1]]$long_name
    } else {
	city_check <- sapply(address$results[[1]]$address_components, has_any_word, words = "postal_town")
	if (length(address$results[[1]]$address_components[city_check])!=0) {
	  city <- address$results[[1]]$address_components[city_check]
	  city <- city[[1]]$long_name
	} else {
	  city <- ""
	}
    }
    street_check <- sapply(address$results[[1]]$address_components, has_any_word, words = "route")
    if (length(address$results[[1]]$address_components[street_check])!=0) {
      street <- address$results[[1]]$address_components[street_check]
	street <- street[[1]]$long_name
    } else {
	street <- ""
    }
  } else {
    city <- revgeo(lon, lat, provider = "bing", API = bing_api_key, output = NULL, item = "city")
  }

  country_check <- sapply(address$results[[1]]$address_components, has_any_word, words = "country")
  if (length(address$results[[1]]$address_components[country_check])!=0) {
    country <- address$results[[1]]$address_components[country_check]
    country <- country[[1]]$long_name
  } else {
    country <- revgeo(lon, lat, provider = "bing", API = bing_api_key, output = NULL, item = "country")
  }
  if (address$results[[1]]$formatted_address != ""){
    address <- address$results[[1]]$formatted_address
  } else {
    address <- revgeo(lon, lat, provider = "bing", API = bing_api_key, output = NULL, item = NULL)
    # address3 <- revgeo(lon, lat, provider = "google", API = google_api_key, output = NULL, item = NULL)
  }  
} else {
  address <- "Unknown"
  city <- "Unknown"
  country <- "Unknown"
}

# Extract the text from the image using Azure Computer Vision API (OCR)
if (file.size(image_path)>4150000){ # needed?
  textOCR <- read_text(vis, raw_vector, detect_orientation = TRUE, language = "en")
} else {
  textOCR <- read_text(vis, image_path, detect_orientation = TRUE, language = "en")
}

# Describe the image using Azure AI:
if (file.size(image_path)>4150000){ # needed?
  text <- analyze(vis, raw_vector, domain = "landmarks", feature_types = "description")$description$captions$text
} else {
  text <- analyze(vis, image_path, domain = "landmarks", feature_types = "description")$description$captions$text
}


# Define a string for the Bing Chat prompt, that will generate the text for the social media post. Feel free to change this to your liking
str <- ""
if (name=="Unknown"){
  str <- paste0("Give me captions with the informative tone for a Story Post on the Platform Instagram about a photo of ",street,", ",city,"; Use at least these hashtags at the end of the captions, but feel free to add more: ",hashtags)
} else{
  if (name==city){
    str <- paste0("Give me captions with the informative tone for a Story Post on the Platform Instagram about a photo of ",text," in ",name,", ",city,"; Use at least these hashtags at the end of the captions, but feel free to add more: ",hashtags)
  } else {
    str <- paste0("Give me captions with the informative tone for a Story Post on the Platform Instagram about a photo of the ",name,", ",city,"; Use at least these hashtags at the end of the captions, but feel free to add more: ",hashtags)
  }
}

} else{
  if (name==city){
    str <- paste0("Write a Story Post for the Platform Instagram about a photo of ",text," in ",name,", ",city,", following these steps: 1- Search for the name of the place on the web and get some information about its history, features and significance; 2- Write a header with the name of the place, the city, the country, the flag emoji and the date of the visit in brackets: (",date,"); 3- Write a dot on a separate line; 4- Add these hashtags ",hashtags," and generate 20 more related to the place, the city, the country, the theme, the camera used (",exif_data$Make," ",exif_data$Model,") and the location itself; Use the informative tone.")
  } else {
    str <- paste0("Write a Story Post for the Platform Instagram about a photo of the ",name,", ",city,", following these steps: 1- Search for the name of the place on the web and get some information about its history, features and significance; 2- Write a header with the name of the place, the city, the country, the flag emoji and the date of the visit in brackets: (",date,"); 3- Write a dot on a separate line; 4- Add these hashtags ",hashtags," and generate 20 more related to the place, the city, the country, the theme, the camera used (",exif_data$Make," ",exif_data$Model,") and the location itself; Use the informative tone.")
  }
}


# Encode the string using URLencode
str_encoded <- URLencode(str, reserved = TRUE)

# Create the Bing Chat URL and open it on the default browser
url <- ""
url <- paste0("https://www.bing.com/search?showconv=1&sendquery=1&q=",str_encoded)

# Open the URL on the default user browser
browseURL(url)
# shell.exec(url)
# browseURL("https://www.bing.com/search?showconv=1&sendquery=1&q=Hello%20Bing")

# Show main results in R Console, which could be used on prompt for Bing Chat
  cat(" Hashtags:     ", hashtags, "\n", "GPS Coordin.: ", lat, ",", lon, "\n", "Landmark Name:", name, "\n", "Landm. Source:", source, "\n", "Text OCR     :", paste(textOCR, collapse = ", "), "\n", "Img. Descript:", paste(text, collapse = ", "), "\n", "Full address: ", address, "\n", "City:         ", city, "\n", "Country:      ", country, "\n", "Camera :      ", exif_data$Make, exif_data$Model, "\n", "Date   :      ", date, "\n")

