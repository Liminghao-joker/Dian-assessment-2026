# 输入随机矩阵， 查看 Encoder 输出
import torch
from Encoder import Encoder

# simple test
config_test_1 = {
    "batch_size": 2,
    "seq_len": 4,
    "d_model": 8,
    "num_heads": 2,
    "d_ff": 16,
    "num_layers": 2
}

# transformer base
config_test_2 = {
    "batch_size": 2,
    "seq_len": 10,
    "d_model": 512,
    "num_heads": 8,
    "d_ff": 2048,
    "num_layers": 6,
    "dropout": 0.1
}

if __name__ == "__main__":
    # Test1 use config_test_1
    # random matrix without embedding
    x = torch.randn(config_test_1["batch_size"], config_test_1["seq_len"], config_test_1["d_model"])
    print("Input shape:", x.shape)
    print("Input:", x)

    # mask
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).unsqueeze(1).repeat(1, config_test_1["seq_len"], 1)  # (batch_size, seq_len, seq_len)
    print("Mask shape:", mask.shape)
    print("Mask:", mask)

    encoder = Encoder(config_test_1["num_layers"], config_test_1["d_model"], config_test_1["num_heads"], config_test_1["d_ff"])

    # first layer test
    encoder_layer = encoder.get_layer(0)

    # Multi-Head Attention
    mha_out, attn_scores = encoder_layer.mha(x, x, x, mask)
    print("X after MHA shape:", mha_out.shape)
    print("X after MHA:", mha_out)
    print("Attention scores shape:", attn_scores.shape)
    print("Attention scores:", attn_scores)

    # Add & Norm 1
    x = encoder_layer.add_norm1(x, lambda z: encoder_layer.mha(z, z, z, mask)[0])
    print("X after Add & Norm 1 shape:", x.shape)
    print("X after Add & Norm 1:", x)

    # Feed Forward
    ffn_out = encoder_layer.ffn(x)
    print("FFN out shape:", ffn_out.shape)
    print("FFN out:", ffn_out)

    # Add & Norm 2
    x = encoder_layer.add_norm2(x, encoder_layer.ffn)
    print("X after Add & Norm 2 shape:", x.shape)
    print("X after Add & Norm 2:", x)

    # 第一轮结束，用该数据继续跑接下来的层
    x_after_layer1 = x
    print("X after first Encoder layer shape:", x_after_layer1.shape)
    print("X after first Encoder layer:", x_after_layer1)

    output = x_after_layer1
    for layer in encoder.layers[1:]:
        output = layer(output, mask)

    x_final = encoder.norm(output)
    print("Final Encoder output shape:", x_final.shape)
    print("Final Encoder output:", x_final)
