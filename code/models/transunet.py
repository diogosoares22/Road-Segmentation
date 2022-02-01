from code.models.unet import Unet
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Add, Conv2D


class TransUNet(Unet):

    def __init__(self, params, load=False, load_params=None):
        super().__init__(params, load=load, load_params=load_params, name="TransUNet")  # Initialize TransUNet

    def transformer_layer(self, previous_layer, num_heads=4, num_keys=8):
        """
        One transformer layer, as described in the paper on TransUNet (https://arxiv.org/pdf/2102.04306.pdf).
        Hyperparameters are different from those of the original architecture, because 1. they are not specified in the
        paper, and 2. we need to use smaller parameters due to our computation limitations.
        @param previous_layer: The previous layer
        @param num_heads: number of attention heads for the MSA layer (i.e. number of times the computation is repeated
        in parallel)
        @param num_keys: size of each attention head for query, key and value (i.e. number of columns for the query, key
        and value matrices)
        @return: transformer layer block
        """

        # The paper describes the following architecture for the transformer layer:
        # - Layer Normalisation
        # - Multihead Self-Attention
        # - Addition of input and MSA output
        # - Layer Normalisation
        # - Multi-Layer Perceptron
        # - Addition of previous addition output and MLP output

        # Some explanations on the layers used:
        # - Multihead Self-Attention (Iris: This is my comprehension of the (self-)attention mechanism, but it may not
        #   be 100% correct)
        #   -  Computes the attention score for each token to each other token (or to each token in the compared set for
        #      cross-attention). Input data is the queries and the keys, while the values are learnt by the network for
        #      computing optimal attention scores.
        #   - In our implementation, we use the MultiHeadAttention layer provided by keras, and make sure to have the
        #     exact same query, key and value dimensions to have self-attention. Having different dimensions results
        #     in cross-attention. Note that the parameter key_dim sets the dimension (columns) for both keys and values.
        # - Multi-Layer Perceptron
        #   - MLPs are a subset of deep neural networks, which comprises only feed-forward neural networks
        #   - Our reference paper for TransUNet does not detail further on which MLP they used. But since we decided to
        #     use a different input shape for our transformer layers, we need to use a convolution. We decided to use a
        #     single Conv2D layer with relu activation function

        layernorm1 = LayerNormalization()(previous_layer)  # Layer normalisation
        msa = MultiHeadAttention(num_heads=num_heads, key_dim=num_keys, value_dim=num_keys)(layernorm1,
                                                                                            layernorm1)  # Multihead Self-Attention
        add1 = Add()([previous_layer, msa])  # Addition
        layernorm2 = LayerNormalization()(add1)  # Layer Normalisation
        mlp = Conv2D(16, kernel_size=2, activation="relu", padding="same")(layernorm2)  # Multi-Layer Perceptron
        add2 = Add()([add1, mlp])  # Addition

        return add2

    def attention_pipeline(self, last_unet_conv, n_transformer_layers=4, n_filters=16):
        """
        Initialize and connect transformer part of the model.
        @param last_unet_conv: Last encoder convolution layer of unet, with shape (16, 16, 16 * n_filters)
        @param n_transformer_layers: Number of transformer layers in the model. Transformer layers are composed of six
        layers performing and processing multi-head self attention
        @param n_filters: factor for the number of filters
        @return: attention layers pipeline
        """

        # The paper adds the following pipeline to the standard UNet version:
        # - Maps the output of the smallest convolution into an array of flattened 2D patches
        # - Performs a linear projection of this embedding into a latent space of size D
        # - Feeds this embedding into transformer layers (the original TransUNet has 16 of them)
        # - Uses a conv2D with 3x3 kernel to upsample the data
        # - Reshapes the obtained attention features into the shape of the ouput of the smallest convolution
        # - Concatenates it with the output of the smallest convolution to feed it back to UNet

        # Some explanations for our implementation:
        # - We decided to completely skip the mapping of the smallest convolution into the array of flattened 2D
        #   patches, because we already are in a small space with 256 (16x16) filters. In the original, paper, they
        #   work with larger images than ours and less convolution layers in UNet, which explains the need to
        #   subdivide in this step, otherwise the attention inputs would be too large (since attention is very
        #   expensive to compute in large spaces)
        # - The paper uses a specific linear projection that enables them to encode spacial information in a
        #   1-dimensional space, and it is unclear for us how exactly they encode this. Therefore, we decide to embed
        #   our representation in a latent space of size (8, 8, 16), so that we keep spacial information. We don't
        #   use any activation function, by definition of linear projection
        # - In the paper, they use 12 transformer layers. However, our model is already very heavy for our computers
        #   and we work with smaller images. So we decided to have only four of those blocks of layers.

        # First, we define the TransUNet pipeline layers
        lin_proj = Conv2D(16, kernel_size=2, strides=2, activation=None)(
            last_unet_conv)  # Linear projection to latent space (8, 8, 16)
        trans = self.transformer_layer(lin_proj)  # Transformer layers
        for i in range(1, n_transformer_layers):
            trans = self.transformer_layer(trans)
        conv3x3 = Conv2D(n_filters * 32, kernel_size=3, activation='relu', padding='same',
                         kernel_initializer='HeNormal')(
            trans)  # Conv3x3 with ReLu to upsample to size (8, 8, n_filters * 32)

        # Next, we connect them properly with layers of UNet
        # - We first create a decoder block with our attention results and the the convolution results obtained
        # - Next, we connect this decoder block to the next decoder block, which is layer 22 of UNet.
        conv_attention_decoder = self.decoder_mini_block(conv3x3, last_unet_conv, n_filters * 16)  # New decoder block
        return conv_attention_decoder

    def init_model(self, n_filters=16, n_transformer_layers=4):
        """
        Initializes the same UNet architecture as the superclass UNet, but adds the attention pipeline to it.
        @param n_filters: factor for the number of filters
        @param n_transformer_layers: Number of transformer layers in the model
        """

        inputs = Input((self.patch_size, self.patch_size, self.num_channels))

        # Encoder blocks
        cblock1 = self.encoder_mini_block(inputs, n_filters, dropout_prob=0, max_pooling=True)
        cblock2 = self.encoder_mini_block(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
        cblock3 = self.encoder_mini_block(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True)
        cblock4 = self.encoder_mini_block(cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=True)
        cblock5 = self.encoder_mini_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

        # Attention pipeline
        attention_pipeline = self.attention_pipeline(cblock5[0], n_transformer_layers=n_transformer_layers,
                                                     n_filters=n_filters)

        # Decoder blocks
        ublock6 = self.decoder_mini_block(attention_pipeline, cblock4[1], n_filters * 8)
        ublock7 = self.decoder_mini_block(ublock6, cblock3[1], n_filters * 4)
        ublock8 = self.decoder_mini_block(ublock7, cblock2[1], n_filters * 2)
        ublock9 = self.decoder_mini_block(ublock8, cblock1[1], n_filters)

        conv9 = Conv2D(n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(
            ublock9)

        conv10 = Conv2D(1, 1, padding='same')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10, name=self.name)

# Notes: Original TransUNet was trained with SGD optimizer, learning rate 0.01, momentum 0.9 and weight decay 1e-4
# But since we modify the architecture a lot for our purposes, original parameters become meaningless anyway
