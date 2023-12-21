# When running first time, please follow this tutorial first:
# https://tensorflow.rstudio.com/install/
#
# After that, run the below command in R:
# remotes::install_github("rstudio/tensorflow")
#
# More information about the prediction model and its results can also be found here:
# https://keras.io/api/applications/

# Load the libraries
library(reticulate)
library(remotes)
library(tensorflow)
library(keras)
library(tfdatasets)
# library(imager)

# Load the image
image_path <- "C:/Users/XXX.jpg"
image <- image_load(image_path, target_size = c(224, 224))

# Preprocess the image
image <- image_to_array(image)
image <- array_reshape(image, c(1, dim(image)))
image <- imagenet_preprocess_input(image)

# Load the pre-trained model
model <- application_resnet50(weights = "imagenet")

# Predict the top 5 classes
preds <- model %>% predict(image)
tags <- imagenet_decode_predictions(preds, top = 5)[[1]]
filtered_tags <- tags[tags$score >= 0.1,]

# Extract the hashtags
hashtags <- paste0("#", tolower(gsub(" ", "", filtered_tags[, 2])))

# Print the hashtags
cat(paste(hashtags, collapse = " "))
