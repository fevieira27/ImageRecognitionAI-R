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

# Load the image - CHANGE HERE!!!
image_path <- "C:/Users/XXXXXXX.jpg"

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
# print(hashtags)

# Print the hashtags only from best results
cat(paste(hashtags, collapse = " "),"\n")

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
