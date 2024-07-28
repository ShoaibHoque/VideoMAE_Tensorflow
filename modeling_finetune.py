from functools import partial
import numpy as np
import tensorflow as tf
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        return drop_path(x, self.drop_prob, training)
    
    def get_config(self):
        config = super().get_config()
        config.update({'drop_prob': self.drop_prob})
        return config

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.keras.activations.gelu, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Dense(hidden_features)
        self.act = act_layer
        self.fc2 = tf.keras.layers.Dense(out_features)
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = tf.keras.layers.Dense(all_head_dim * 3, use_bias=qkv_bias)
        self.q_bias = tf.Variable(tf.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = tf.Variable(tf.zeros(all_head_dim)) if qkv_bias else None

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def call(self, x, training=False):
        B, N, C = tf.shape(x)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = tf.concat([self.q_bias, tf.zeros_like(self.v_bias), self.v_bias], axis=0)
        qkv = tf.linalg.matmul(x, self.qkv.weights[0]) + qkv_bias
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        q, k, v = tf.unstack(qkv, axis=2)

        q = q * self.scale
        attn = tf.einsum('bhqd,bhkd->bhqk', q, k)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.einsum('bhqk,bhvd->bhqd', attn, v)
        x = tf.reshape(x, [B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

class Block(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., init_values=None, act_layer=tf.keras.activations.gelu, norm_layer=tf.keras.layers.LayerNormalization, attn_head_dim=None):
        super(Block, self).__init__()
        self.norm1 = norm_layer(epsilon=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.keras.layers.Layer()
        self.norm2 = norm_layer(epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = tf.Variable(init_values * tf.ones((dim)), trainable=True)
            self.gamma_2 = tf.Variable(init_values * tf.ones((dim)), trainable=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def call(self, x, training=False):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)), training=training)
            x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)), training=training)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)), training=training)
        return x

class PatchEmbed(tf.keras.layers.Layer):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = tf.keras.layers.Conv3D(filters=embed_dim, kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]), strides=(self.tubelet_size, patch_size[0], patch_size[1]), padding='valid')

    def call(self, x, **kwargs):
        B, T, H, W, C = tf.shape(x)
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(x, [B, -1, x.shape[-1]])
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    
    return tf.convert_to_tensor(sinusoid_table, dtype=tf.float32)

class VisionTransformer(tf.keras.Model):
    """ Vision Transformer with support for patch or hybrid CNN input stage """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, fc_drop_rate=0., drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=tf.keras.layers.LayerNormalization, init_values=0., use_learnable_pos_emb=False, init_scale=0., all_frames=16, tubelet_size=2, use_checkpoint=False, use_mean_pooling=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.add_weight('pos_embed', shape=[1, num_patches, embed_dim], initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.cls_token = self.add_weight('cls_token', shape=[1, 1, embed_dim], initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.pos_drop = tf.keras.layers.Dropout(drop_rate)
        self.use_checkpoint = use_checkpoint

        dpr = [x.numpy() for x in tf.linspace(0., drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], init_values=init_values, norm_layer=norm_layer) for i in range(depth)]
        self.norm = norm_layer(epsilon=1e-6)

        self.use_mean_pooling = use_mean_pooling
        self.fc_norm = norm_layer(epsilon=1e-6) if use_mean_pooling else None
        self.head_drop = tf.keras.layers.Dropout(fc_drop_rate)
        self.head = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02), trainable=True) if num_classes > 0 else tf.keras.layers.Layer()

        self.init_scale = init_scale
        self.apply(self._init_weights)

        if not use_learnable_pos_emb:
            self.pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[2])
            self.pos_embed = tf.Variable(tf.convert_to_tensor(self.pos_embed, dtype=tf.float32), trainable=False)

    def apply(self, fn):
        for layer in self.layers:
            fn(layer)
    
    def _init_weights(self, layer):
        if isinstance(layer, tf.keras.layers.Dense):
            trunc_normal_(layer.kernel, std=self.init_scale)
            if layer.bias is not None:
                tf.keras.initializers.Zeros()(layer.bias)
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            tf.keras.initializers.Zeros()(layer.beta)
            tf.keras.initializers.Ones()(layer.gamma)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            trunc_normal_(layer.kernel, std=self.init_scale)
            if layer.bias is not None:
                tf.keras.initializers.Zeros()(layer.bias)

    def forward_features(self, x, training=False):
        B = tf.shape(x)[0]
        x = self.patch_embed(x)
        cls_tokens = tf.tile(self.cls_token, [B, 1, 1])
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embed
        x = self.pos_drop(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.norm(x)
        if self.use_mean_pooling:
            return self.fc_norm(tf.reduce_mean(x[:, 1:], axis=1))
        else:
            return x[:, 0]

    def call(self, x, training=False):
        x = self.forward_features(x, training=training)
        x = self.head_drop(x, training=training)
        x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            out_dict[k] = v.reshape([v.shape[0], 3, patch_size, patch_size])
        else:
            out_dict[k] = v
    return out_dict

@register_model
def vit_base_patch16_224_TransReID(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(tf.keras.layers.LayerNormalization, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_patch16_224_TransReID(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(tf.keras.layers.LayerNormalization, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_patch16_224_TransReID(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(tf.keras.layers.LayerNormalization, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
