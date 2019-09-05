import os, argparse
import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers.merge import concatenate, multiply
import matplotlib.pyplot as plt


def main(dataset_nm='mnist', batch_size=64, epochs=20000,
         save_interval=100):
    cgan = ConditionalGAN(dataset_nm)
    cgan.train(batch_size, epochs, save_interval)
    cgan.plot_hist()


class ConditionalGAN:
    def __init__(self, dataset_nm='mnist'):
        for dir_nm in ('generated_images', 'history', 'gif'):
            os.makedirs(f'figures/{dir_nm}/', exist_ok=True)
        self.dataset_nm = dataset_nm
        self.X_train, self.y_train = self.load_data()
        self.n_data = self.X_train.shape[0]
        self.img_shape = self.X_train.shape[1:]
        self.img_row, self.img_col, self.channels = self.img_shape
        self.n_classes = self.y_train.shape[1]
        self.latent_dim = 100
        self.last_epoch = 0
        self.losses = {}

        d_optimizer = Adam(lr=2e-4, beta_1=0.1)
        g_optimizer = Adam(lr=2e-4, beta_1=0.5)

        # Build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=d_optimizer,
            metrics=['acc'])
        print('discriminator model')
        self.discriminator.summary()

        # Build generator and combine with discriminator.
        self.generator = self.build_generator()
        self.combined = self.build_combined()
        # While trianing generator, parameters in discriminator must be fixed.
        self.discriminator.trainable = False
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=g_optimizer,
            metrics=['acc'])
        print('generator model')
        self.combined.summary()

    def load_data(self):
        ds_list = {'mnist': mnist, 'fashion_mnist': fashion_mnist,
                   'cifar10': cifar10}
        ds = ds_list[self.dataset_nm]
        (X_train, y_train), (X_test, y_test) = ds.load_data()
        X_train = np.concatenate([X_train, X_test])
        y_train = np.concatenate([y_train, y_test]).reshape(-1,)
        # If images are black and white, add channel axis.
        if self.dataset_nm in ('mnist', 'fashion_mnist'):
            X_train = X_train[:, :, :, np.newaxis]
        # Scale imgs -1.0 ~ 1.0
        X_train = X_train / 127.5 - 1.
        # Encode labels into binary vectors.
        n_classes = len(np.unique(y_train))
        y_train = np.eye(n_classes)[y_train]
        return X_train, y_train

    def build_generator(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.latent_dim + self.n_classes))
        model.add(LeakyReLU())
        r, c = int(self.img_row / 4), int(self.img_col / 4)
        model.add(Dense(128 * r * c))
        model.add(BatchNormalization(axis=-1, momentum=0.8))
        model.add(LeakyReLU())
        model.add(Reshape((r, c, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, 5, padding='same'))
        # If Deconv, use Conv2DTranspose like this.
        # model.add(Conv2DTranspose(128, 5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, 5, padding='same',
                         activation='tanh'))

        z = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.n_classes,))
        model_input = concatenate([z, label])
        print('model_input shape :', model_input.shape)
        img = model(model_input)
        return Model([z, label], img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, 5, padding='same',
                         input_shape=(self.img_row, self.img_col,
                                      self.channels + self.n_classes)))
        model.add(LeakyReLU())
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Conv2D(128, 5, strides=2,
                         kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(256, kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(1,kernel_initializer='he_normal',
                        activation='sigmoid'))

        img = Input(shape=self.img_shape)
        label = Input(shape=(self.n_classes,))
        label_unpooled = UpSampling2D(self.img_shape[:2])(
            Reshape((1, 1, self.n_classes))(label))
        model_input = concatenate([img, label_unpooled])
        valid = model(model_input)
        return Model([img, label], valid)

    def build_combined(self):
        z = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.n_classes,))
        img = self.generator([z, label])
        valid = self.discriminator([img, label])
        return Model([z, label], valid)

    def train(self, batch_size=128, epochs=1000, save_interval=100):
        for epoch in range(self.last_epoch, self.last_epoch + epochs):
            loss = {}
            # ----------------------
            # Train Discriminator
            # ----------------------
            ## Select a random batch of images.
            half_batch = int(batch_size/2)
            idx = np.random.randint(0, self.n_data, half_batch)
            imgs, labels = self.X_train[idx], self.y_train[idx]

            ## Sample noises as generator input.
            noises = np.random.normal(0, 1, (half_batch, self.latent_dim))
            gen_imgs = self.generator.predict([noises, labels])

            ## Train the discriminator.
            d_loss_real = self.discriminator.train_on_batch(
                [imgs, labels], np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(
                [gen_imgs, labels], np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            #if d_loss[1] > 0.8:
            #    self.discriminator.trainable = False
            #else:
            #    self.discriminator.trainable = True

            # ----------------------
            # Train Generator
            # ----------------------
            ## Sample noises and condition on labels.
            noises = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = np.eye(self.n_classes)[
                np.random.randint(0, 10, batch_size)]
            ## Train the generator.
            g_loss = self.combined.train_on_batch(
                [noises, sampled_labels], np.ones((batch_size, 1)))

            # Append losses and change the last epoch
            self.losses[epoch] = {'D_loss': d_loss[0],
                                  'D_acc': d_loss[1],
                                  'G_loss': g_loss[0]}
            self.last_epoch = epoch

            if epoch % save_interval == 0:
                print(f"epoch: {epoch} {self.losses[epoch]}")
                self.sample_imgs(True)

    def sample_imgs(self, save=True):
        row, col = 2, 5
        noises = np.random.normal(0, 1, (row * col, self.latent_dim))
        sampled_labels = np.eye(self.n_classes)[np.arange(0, 10)]
        gen_imgs = self.generator.predict([noises, sampled_labels])
        if self.dataset_nm in ('mnist', 'fashion_mnist'):
            gen_imgs = gen_imgs[:, :, :, 0]
            cmap = 'gray'
        else:
            cmap = None

        fig, ax = plt.subplots(row, col, figsize=(5, 4))
        for r in range(row):
            for c in range(col):
                label = r * col + c
                img = gen_imgs[label, :, :]
                ax[r, c].set_title(f'label={label}')
                ax[r, c].imshow(img, cmap=cmap)
                ax[r, c].axis('off')
        plt.suptitle(f'epoch={self.last_epoch}', size=20)
        if save:
            fig.savefig('figures/generated_images/'
                        f'cgan_{self.dataset_nm}_{self.last_epoch}.png')
        plt.show()
        plt.close()

    def plot_hist(self, save=True):
        epoch = self.losses.keys()
        d_acc_hist = [loss['D_acc'] for loss in self.losses.values()]
        d_loss_hist = [loss['D_loss'] for loss in self.losses.values()]
        g_loss_hist = [loss['G_loss'] for loss in self.losses.values()]

        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax1.set_xlabel('epoch', fontsize=25)
        ax1.set_ylabel('Loss', fontsize=25)
        ax1.plot(epoch, d_loss_hist, label='Discriminator')
        ax1.plot(epoch, g_loss_hist, label='Generator')
        ax1.tick_params(axis='y', labelsize=15)
        ax1.legend(fontsize=15, loc='upper left')

        ax2 = ax1.twinx()
        color = 'b'
        ax2.plot(epoch, d_acc_hist, color=color,
                 label='Discriminator Accuracy', linewidth=0.1)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
        ax2.legend(fontsize=15, loc='upper right')
        plt.show()
        if save:
            plt.savefig(f'figures/hist/cgan_{self.dataset_nm}_hist.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset_nm',
                        choices=['minst', 'fashion_mnist', 'cifar10'])
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-si', '--save_interval', type=int)
    args = parser.parse_args()
    dataset_nm = args.dataset_nm
    batch_size=args.batch_size
    epochs = args.epochs
    save_interval = args.save_interval
    main(dataset_nm, batch_size, epochs, save_interval)
