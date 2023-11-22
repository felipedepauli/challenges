# Author: Felipe Camargo de Pauli
# Date  : 15/11/2023
# This was made in VIM using the LABIC's cluster

import tensorflow as tf

print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
print("Tensorflow Version:", tf.__version__)

# Set the data path
#train_data_path = "./10_food_classes_all_data/train"
#test_data_path  = "./10_food_classes_all_data/test"
train_data_path = "./food-101/train"
test_data_path  = "./food-101/test"

# Create the image generator without data augmentation (only rescaling to the CNN's input size)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#- train_data_gen = ImageDataGenerator(rescale=1/255.)   
test_data_gen  = ImageDataGenerator(rescale=1/255.)

#- # Generate both the train_data and the test_data with the generator
#- train_data = train_data_gen.flow_from_directory(
#-         directory       = train_data_path,
#-         target_size     = (224, 224),
#-         batch_size      = 32,
#-         class_mode      = "categorical",
#-         seed            = 42
#- )
test_data = test_data_gen.flow_from_directory(
        directory       = test_data_path,
        target_size     = (224, 224),
        batch_size      = 1,
        class_mode      = "categorical",
        seed            = 42
)
#- 
# Create the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, BatchNormalization
#- 
#- model_1 = Sequential([
#- 
#-     Conv2D(
#-         filters     = 10,
#-         kernel_size = (3, 3),
#-         input_shape = (224, 224, 3),
#-         activation  = "relu"
#-     ),
#-     Conv2D(10, 3, activation= "relu"),
#-     MaxPool2D(),
#- 
#-     Conv2D(10, 3, activation= "relu"),
#-     Conv2D(10, 3, activation= "relu"),
#-     MaxPool2D(),
#- 
#-     Flatten(),
#- 
#-     Dense(10, activation="softmax")
#- 
#- ])
#- 
#- # Compile the model
#- model_1.compile(
#-         optimizer = Adam(),
#-         metrics   = ["accuracy"],
#-         loss      = "categorical_crossentropy"
#- )
#- 
#- # Let's look at the architecture
#- model_1.summary()
#- 
#- # Time to work!
#- all_histories = []
#- 
#- history_1 = model_1.fit(
#-          train_data,
#-          epochs           = 1,
#-          steps_per_epoch  = len(train_data),
#-          validation_data  = test_data,
#-          validation_steps = len(test_data)
#- )
#- 
#- all_histories.append(history_1)
#- 
#- print("------------------------------")
#- print("Evaluation (Test_Directory)")
#- model_1.evaluate(test_data)
#- print("------------------------------")

# Analyse the results
import plotext as plt

# We're going to create a function that runs on terminal
# since the Labic does not give me access through VSCode
def plot_results(history):
    history_dict = history.history

    epochs = range(1, len(history_dict['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss and Loss validation plot
    ax1.plot(epochs, history_dict['loss'], label="Loss")
    ax1.plot(epochs, history_dict['val_loss'], label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy and Accuracy validation plot
    ax2.plot(epochs, history_dict['accuracy'], label="Accuracy")
    ax2.plot(epochs, history_dict['val_accuracy'], label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()


#- plot_results(history_1)

# It was verified there is a overfitting.
# The training got a great result in accuracy data, but when we tried to use in validation data... nothing good.
# We got a 0.8840 accuracy in training set, and 0.2584 in validation set.
# To decrease that value, we could do the following:
# 1. Get more data; (it's not possible)
# 2. Simplify our architecture with less filters or layers (we could lose the training accuracy)
# 3. Data augmentation (THAT'S IT!)

train_data_gen = ImageDataGenerator(
    rescale             = 1/255.,       # Normalize the data to fit between 0 and 1 (float)
    rotation_range      = 0.3,
    width_shift_range   = 0.2,
    height_shift_range  = 0.2,
    zoom_range          = 0.1,
    horizontal_flip     = True,
    fill_mode           = "nearest"
)

train_data_augmented = train_data_gen.flow_from_directory(
    directory           = train_data_path,
    class_mode          = "categorical",
    batch_size          = 4,
    seed                = 42,
    target_size         = (224, 224),
    shuffle             = True
)

#- history_1_2 = model_1.fit(
#-     train_data_augmented,
#-     epochs              = 1,
#-     steps_per_epoch     = len(train_data),
#-     validation_data     = test_data,
#-     validation_steps    = len(test_data)
#- )
#- 
#- all_histories.append(history_1_2)
#- 
#- print("------------------------------")
#- model_1.evaluate(test_data)
#- print("------------------------------")
#- plot_results(history_1_2)
#- 

# It's better! But we ought get a better model.
# I think it's time to tunning with the correct hyperparameters!
# The best way to get the best hyperparameters is using the optuna.

from tensorflow.keras.layers import Dropout

#- def build_model(hp):
#-     model = Sequential()
#- 
#-     for i in range(hp.Int('conv_blocks', 1, 5, default=2)):
#-         model.add(Conv2D(
#-             filters     = hp.Int(f'filters_{i}', 16, 256, step=16),
#-             kernel_size = hp.Choice(f'kernel_size_{i}', [3, 5]),
#-             activation  = "relu"
#-         ))
#-         model.add(Conv2D(
#-             filters     = hp.Int(f'filters_{i}', 16, 256, step=16),
#-             kernel_size = hp.Choice(f'kernel_size_{i}', [3, 5]),
#-             activation  = "relu"
#-         ))
#-         model.add(MaxPool2D(2))
#-     
#-     model.add(Flatten())
#- 
#-     model.add(Dense(
#-         units=hp.Int("unit", min_value=32, max_value=512, step=32),
#-         activation="relu"
#-     ))
#- 
#-     # Dropout to avoid overfitting
#-     model.add(Dropout(
#-         rate=hp.Float('dropout', min_value=0.0,max_value=0.5, default=0.25, step=0.05)
#-     ))
#- 
#-     # Output layer with the 10 classes with their probabilities
#-     # The sum of all probabilites is 1
#-     model.add(Dense(10, activation="softmax"))
#- 
#-     model.compile(
#-         optimizer = tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
#-         loss = 'categorical_crossentropy',
#-         metrics = ['accuracy']
#-     )
#- 
#-     return model
#- 

from keras_tuner.tuners import RandomSearch
#- 
#- tuner = RandomSearch(
#-     build_model,
#-     objective       = "val_accuracy",
#-     max_trials      = 10,
#-     executions_per_trial = 1,
#-     directory       = "ea",
#-     project_name    = "cnn_final"
#- )
#- 
#- tuner.search(
#-     train_data_augmented,
#-     epochs          = 10,
#-     validation_data = test_data
#- )
        
#- def build_model(hp):
#-     model = Sequential()
#- 
#-     model.add(Conv2D(
#-         filters     = 32, 
#-         kernel_size = 3,
#-     ))
#-     model.add(BatchNormalization())
#-     model.add(Activation("relu"))
#- 
#-     model.add(Conv2D(
#-         filters     = 32,
#-         kernel_size = 3,
#-     ))
#-     model.add(BatchNormalization())
#-     model.add(Activation("relu"))
#-     model.add(MaxPool2D())
#-    
#-     model.add(Conv2D(
#-         filters = 64,
#-         kernel_size = 3,
#-         activation  = "relu"
#-     ))
#-     model.add(MaxPool2D())
#-     model.add(Conv2D(
#-         filters = hp.Int(f'filters_end', 128, 512, step=32),
#-         kernel_size = 3,
#-         activation  = "relu"
#-     ))
#-     model.add(MaxPool2D())
#-  
#-     model.add(Flatten())
#- 
#-     model.add(Dense(
#-         units=hp.Int("unit", min_value=256, max_value=512, step=32),
#-         activation="relu"
#-     ))
#-     # Dropout to avoid overfitting
#-     model.add(Dropout(
#-         rate=0.45
#-     ))
#- 
#-     # Output layer with the 10 classes with their probabilities
#-     # The sum of all probabilites is 1
#-     model.add(Dense(10, activation="softmax"))
#- 
#-     model.compile(
#-         optimizer = Adam(),
#-         loss = 'categorical_crossentropy',
#-         metrics = ['accuracy']
#-     )
#-     return model
#- 
#- tuner = RandomSearch(
#-     build_model,
#-     objective       = "val_accuracy",
#-     max_trials      = 200,
#-     executions_per_trial = 1,
#-     directory       = "ea",
#-     project_name    = "cnn_tunning_final"
#- )
#- 
#- tuner.search(
#-     train_data_augmented,
#-     epochs          = 15,
#-     validation_data = test_data
#- )
#- 



model_final = Sequential([

    Conv2D(32, 3, input_shape=(224, 224, 3)),
    BatchNormalization(),
    Activation("relu"),
#    Conv2D(64, 3),
#    BatchNormalization(),
#    Activation("relu"),
    MaxPool2D(),

#    Conv2D(128, 3, activation="relu"),
    Conv2D(64, 3, activation="relu"),
    MaxPool2D(),
    
#    Conv2D(256, 3, activation="relu"),
    Conv2D(128, 3, activation="relu"),
    MaxPool2D(),

    Flatten(),
    
    Dense(units=400, activation="relu"),
    Dropout(rate=0.4),

#    Dense(units=200, activation="relu"),
#    Dropout(rate=0.4),

    Dense(101, activation="softmax")

])

from tensorflow.keras.callbacks import EarlyStopping

# Configuração do Early Stopping
early_stopping = EarlyStopping(
    monitor     = 'val_accuracy', # Qual métrica monitorar
    patience    = 25,             # Quantas épocas esperar após a última melhoria
    min_delta   = 0.0005,         # Mudança mínima considerada como uma melhoria
    mode        = 'max',          # Minimização da métrica (para 'loss') ou maximização (para 'accuracy')
    verbose     = 1,              # Mostrar mensagens quando o early stopping é acionado
    restore_best_weights = True   # Restaura os pesos do modelo para o estado da melhor época
)

model_final.compile(
    optimizer   = Adam(1e-4),
    loss        = "categorical_crossentropy",
    metrics     = ["accuracy"]
)

history_final = model_final.fit(
    train_data_augmented,
    epochs                  = 200,
    steps_per_epoch         = len(train_data_augmented),
    validation_data         = test_data,
    validation_steps        = len(test_data),
    callbacks               = [early_stopping]
)

model_final.save("final_model_2.h5")
