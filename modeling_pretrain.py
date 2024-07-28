import tensorflow as tf
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model

__all__ = [
    'pretrain_videomae_small_patch16_224',
    'pretrain_videomae_base_patch16_224',
    'pretrain_videomae_large_patch16_224',
    'pretrain_videomae_huge_patch16_224',
]


def trunc_normal_(tensor, mean=0., std=1.):
    initializer = tf.keras.initializers.TruncatedNormal(mean=mean, stddev=std)
    tensor.assign(initializer(tensor.shape))


class PretrainVisionTransformerEncoder(tf.keras.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=tf.keras.layers.LayerNormalization, init_values=None, tubelet_size=2,
                 use_checkpoint=False, use_learnable_pos_emb=False):
        super(PretrainVisionTransformerEncoder, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = self.add_weight("pos_embed", shape=[1, num_patches + 1, embed_dim], initializer='zeros')
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = tf.linspace(0.0, drop_path_rate, depth).numpy()
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                             init_values=init_values) for i in range(depth)]
        self.norm = norm_layer()
        self.head = tf.keras.layers.Dense(num_classes) if num_classes > 0 else tf.identity

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotUniform()
                if layer.bias is not None:
                    layer.bias_initializer = tf.keras.initializers.Zeros()
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                layer.beta_initializer = tf.keras.initializers.Zeros()
                layer.gamma_initializer = tf.keras.initializers.Ones()

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = tf.keras.layers.Dense(num_classes) if num_classes > 0 else tf.identity

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        x = x + tf.cast(self.pos_embed, x.dtype)
        B, _, C = x.shape
        x_vis = tf.reshape(x[~mask], (B, -1, C))
        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)
        return x_vis

    def call(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(tf.keras.Model):
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=tf.keras.layers.LayerNormalization, init_values=None, num_patches=196, tubelet_size=2,
                 use_checkpoint=False):
        super(PretrainVisionTransformerDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = tf.linspace(0.0, drop_path_rate, depth).numpy()
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                             init_values=init_values) for i in range(depth)]
        self.norm = norm_layer()
        self.head = tf.keras.layers.Dense(num_classes) if num_classes > 0 else tf.identity

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotUniform()
                if layer.bias is not None:
                    layer.bias_initializer = tf.keras.initializers.Zeros()
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                layer.beta_initializer = tf.keras.initializers.Zeros()
                layer.gamma_initializer = tf.keras.initializers.Ones()

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = tf.keras.layers.Dense(num_classes) if num_classes > 0 else tf.identity

    def call(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x


class PretrainVisionTransformer(tf.keras.Model):
    def __init__(self, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, encoder_embed_dim=768,
                 encoder_depth=12, encoder_num_heads=12, decoder_num_classes=1536, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=tf.keras.layers.LayerNormalization, init_values=0., use_learnable_pos_emb=False,
                 use_checkpoint=False, tubelet_size=2, num_classes=0, in_chans=0):
        super(PretrainVisionTransformer, self).__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint, use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, num_patches=self.encoder.patch_embed.num_patches, num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = tf.keras.layers.Dense(decoder_embed_dim, use_bias=False)

        self.mask_token = self.add_weight("mask_token", shape=[1, 1, decoder_embed_dim], initializer='zeros')

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotUniform()
                if layer.bias is not None:
                    layer.bias_initializer = tf.keras.initializers.Zeros()
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                layer.beta_initializer = tf.keras.initializers.Zeros()
                layer.gamma_initializer = tf.keras.initializers.Ones()

    def call(self, x, mask):
        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        B, _, C = x_vis.shape
        _, N = mask.shape
        len_keep = N - tf.reduce_sum(mask, axis=1)
        noise = tf.random.uniform((B, N), 0.0, 1.0)
        ids_shuffle = tf.argsort(noise, axis=1)
        ids_restore = tf.argsort(ids_shuffle, axis=1)
        mask_tokens = tf.repeat(self.mask_token, repeats=B * tf.reduce_sum(mask), axis=0)
        mask_tokens = tf.reshape(mask_tokens, (B, -1, C))

        x_full = tf.concat([x_vis, mask_tokens], axis=1)
        x_full = tf.gather(x_full, ids_restore, batch_dims=1)
        x_full = x_full + tf.cast(self.pos_embed, x_full.dtype)
        x = self.decoder(x_full, self.encoder.patch_embed.num_patches - len_keep)
        return x


@register_model
def pretrain_videomae_small_patch16_224(**kwargs):
    model = PretrainVisionTransformer(
        img_size=224, patch_size=16, encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6, mlp_ratio=4,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, decoder_embed_dim=192, decoder_depth=4,
        decoder_num_heads=3, init_values=0.1, tubelet_size=2, **kwargs)
    return model


@register_model
def pretrain_videomae_base_patch16_224(**kwargs):
    model = PretrainVisionTransformer(
        img_size=224, patch_size=16, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, mlp_ratio=4,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, decoder_embed_dim=384, decoder_depth=4,
        decoder_num_heads=6, init_values=0.1, tubelet_size=2, **kwargs)
    return model


@register_model
def pretrain_videomae_large_patch16_224(**kwargs):
    model = PretrainVisionTransformer(
        img_size=224, patch_size=16, encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16, mlp_ratio=4,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, decoder_embed_dim=512, decoder_depth=8,
        decoder_num_heads=16, init_values=0.1, tubelet_size=2, **kwargs)
    return model


@register_model
def pretrain_videomae_huge_patch16_224(**kwargs):
    model = PretrainVisionTransformer(
        img_size=224, patch_size=16, encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16, mlp_ratio=4,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, decoder_embed_dim=640, decoder_depth=8,
        decoder_num_heads=16, init_values=0.1, tubelet_size=2, **kwargs)
    return model
