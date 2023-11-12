import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten, UpSampling2D, LeakyReLU, Dense, Input, add
from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define blocks to build the generator
def residual_block(input_tensor):
    res_model = Conv2D(64, (3,3), padding="same")(input_tensor)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1,2])(res_model)
    res_model = Conv2D(64, (3,3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    return add([input_tensor, res_model])

def upscale_block(input_tensor):
    up_model = Conv2D(256, (3,3), padding="same")(input_tensor)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    return up_model

# Generator model
def build_generator(input_shape, num_res_blocks):
    gen_input = Input(shape=input_shape)
    gen_model = Conv2D(64, (9,9), padding="same")(gen_input)
    gen_model = PReLU(shared_axes=[1,2])(gen_model)
    temp_model = gen_model

    for _ in range(num_res_blocks):
        gen_model = residual_block(gen_model)

    gen_model = Conv2D(64, (3,3), padding="same")(gen_model)
    gen_model = BatchNormalization(momentum=0.5)(gen_model)
    gen_model = add([gen_model, temp_model])
    gen_model = upscale_block(gen_model)
    gen_model = upscale_block(gen_model)
    gen_output = Conv2D(3, (9,9), padding="same")(gen_model)

    return Model(inputs=gen_input, outputs=gen_output)

# Discriminator block
def discriminator_block(input_tensor, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3,3), strides=strides, padding="same")(input_tensor)
    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)
    disc_model = LeakyReLU(alpha=0.2)(disc_model)
    return disc_model

# Discriminator model
def build_discriminator(input_shape):
    disc_input = Input(shape=input_shape)
    df = 64

    d1 = discriminator_block(disc_input, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)

    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(inputs=disc_input, outputs=validity)

# VGG model for feature extraction
def build_vgg(input_shape):
    vgg_model = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    return Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[10].output)

# Combined GAN model
def build_gan(generator, discriminator, vgg, input_lr_shape, input_hr_shape):
    input_lr = Input(shape=input_lr_shape)
    input_hr = Input(shape=input_hr_shape)

    gen_img = generator(input_lr)
    gen_features = vgg(gen_img)

    discriminator.trainable = False
    validity = discriminator(gen_img)

    combined_model = Model(inputs=[input_lr, input_hr], outputs=[validity, gen_features])
    combined_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
    return combined_model

# Preprocess and load data
def preprocess_load_data(low_res_folder, high_res_folder, number_of_images):
    low_res_images, high_res_images = [], []

    for img_name in os.listdir(low_res_folder)[:number_of_images]:
        img = cv2.imread(os.path.join(low_res_folder, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        low_res_images.append(img)

    for img_name in os.listdir(high_res_folder)[:number_of_images]:
        img = cv2.imread(os.path.join(high_res_folder, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        high_res_images.append(img)

    return np.array(low_res_images) / 255.0, np.array(high_res_images) / 255.0

def train_gan(generator, discriminator, gan_model, vgg, lr_train, hr_train, epochs, batch_size):
    fake_labels = np.zeros((batch_size, 1))
    real_labels = np.ones((batch_size, 1))

    for epoch in range(epochs):
        g_losses, d_losses = [], []

        for batch_index in tqdm(range(len(lr_train) // batch_size)):
            start_idx = batch_index * batch_size
            end_idx = start_idx + batch_size

            lr_imgs = lr_train[start_idx:end_idx]
            hr_imgs = hr_train[start_idx:end_idx]

            fake_imgs = generator.predict(lr_imgs)

            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            discriminator.trainable = False
            image_features = vgg.predict(hr_imgs)
            g_loss = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_labels, image_features])

            g_losses.append(g_loss)
            d_losses.append(d_loss)

        g_loss_epoch = np.mean(g_losses, axis=0)
        d_loss_epoch = np.mean(d_losses, axis=0)
        print(f"Epoch: {epoch + 1}/{epochs}, Generator Loss: {g_loss_epoch}, Discriminator Loss: {d_loss_epoch}")

        if (epoch + 1) % 10 == 0:
            generator.save(f"generator_epoch_{epoch + 1}.h5")


def test_gan(generator, lr_test, hr_test):
    # Select a random index in the test dataset
    ix = np.random.randint(0, len(lr_test), 1)
    src_image, tar_image = lr_test[ix], hr_test[ix]

    # Generate the high-resolution output from the low-resolution input
    gen_image = generator.predict(src_image)

    # Plot the low-resolution, generated high-resolution, and original high-resolution images
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Low Resolution')
    plt.imshow(src_image[0])
    plt.subplot(232)
    plt.title('Generated High Resolution')
    plt.imshow(gen_image[0])
    plt.subplot(233)
    plt.title('Original High Resolution')
    plt.imshow(tar_image[0])
    plt.show()


# Main code
low_res_images, high_res_images = preprocess_load_data("data/low_images", "data/high_images", 5000)
lr_train, lr_test, hr_train, hr_test = train_test_split(low_res_images, high_res_images, test_size=0.33, random_state=42)

generator = build_generator(lr_train.shape[1:], 16)
discriminator = build_discriminator(hr_train.shape[1:])
vgg = build_vgg(hr_train.shape[1:])
vgg.trainable = False

gan_model = build_gan(generator, discriminator, vgg, lr_train.shape[1:], hr_train.shape[1:])
train_gan(generator, discriminator, gan_model, vgg, lr_train, hr_train, epochs=5, batch_size=1)
test_gan(generator, lr_test, hr_test)
