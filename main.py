# main.py
#this is for viewing the model parameters
from model_building_and_training import build_model
# Get number of classes from training generator
num_classes = len(train_generator.class_indices)

# Build the model
model = build_model(num_classes)



# Print the summary to check the model parameters
model.summary()