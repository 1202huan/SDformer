import matplotlib.pyplot as plt
import os
import seaborn
import torch



class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_accuracies(accuracy_list, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{folder}/training-graph.png')
    plt.clf()


def plot_attention(model, layers, folder):
    # 检查模型是否有禁用注意力可视化的标志
    if hasattr(model, 'no_attention_visualization') and model.no_attention_visualization:
        print(f"跳过模型 {model.name} 的注意力可视化")
        return
        
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    for layer in range(layers):  # layers
        # 检查模型类型并相应地调整图表布局
        if hasattr(model, 'transformer_encoder2'):
            # 原始DTAAD模型有两个编码器
            fig, (axs, axs1) = plt.subplots(1, 2, figsize=(10, 4))
            
            # 获取第一个编码器的注意力分数
            if hasattr(model.transformer_encoder1.layers[layer], 'self_attn'):
                # 原始TransformerEncoderLayer方式
                att1 = model.transformer_encoder1.layers[layer].att[0].data.cpu()
            else:
                # SparseTransformerEncoderLayer方式 - 取第一个batch，第一个head
                att1 = model.transformer_encoder1.layers[layer].att[0, 0].data.cpu()
            
            # 获取第二个编码器的注意力分数
            if hasattr(model.transformer_encoder2.layers[layer], 'self_attn'):
                # 原始TransformerEncoderLayer方式
                att2 = model.transformer_encoder2.layers[layer].att[0].data.cpu()
            else:
                # SparseTransformerEncoderLayer方式 - 取第一个batch，第一个head
                att2 = model.transformer_encoder2.layers[layer].att[0, 0].data.cpu()
            
            heatmap = seaborn.heatmap(att1, ax=axs)
            heatmap.set_title("Local_attention", fontsize=10)
            heatmap = seaborn.heatmap(att2, ax=axs1)
            heatmap.set_title("Global_attention", fontsize=10)
        else:
            # SDfomer模型只有一个编码器
            fig, axs = plt.subplots(figsize=(8, 6))
            
            # 获取唯一编码器的注意力分数
            if hasattr(model.transformer_encoder1.layers[layer], 'self_attn'):
                # 原始TransformerEncoderLayer方式
                att = model.transformer_encoder1.layers[layer].att[0].data.cpu()
            else:
                # SparseTransformerEncoderLayer方式 - 取第一个batch，第一个head
                att = model.transformer_encoder1.layers[layer].att[0, 0].data.cpu()
            
            heatmap = seaborn.heatmap(att, ax=axs)
            heatmap.set_title("Transformer Attention", fontsize=10)
        
    # 保存热力图
    heatmap.get_figure().savefig(f'plots/{folder}/attention-score.png')
    plt.clf()


def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage * 100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]


def getresults2(df, result):  # all dims-sum & mean
    results2, df1, df2 = {}, df.sum(), df.mean()
    for a in ['FN', 'FP', 'TP', 'TN']:
        results2[a] = df1[a]
    for a in ['precision', 'recall', 'ROC/AUC']:
        results2[a] = df2[a]
    results2['f1_mean'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
    return results2


def plot_differential_attention(model, folder):
    """
    为SDfomer模型绘制差分注意力热力图
    
    参数:
    model: SDfomer模型实例
    folder: 保存热力图的文件夹名称
    """
    # 检查模型是否有禁用注意力可视化的标志
    if hasattr(model, 'no_attention_visualization') and model.no_attention_visualization:
        print(f"跳过模型 {model.name} 的差分注意力可视化")
        return
    
    # 创建保存目录
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    
    # 获取一个样本输入进行前向传播
    # 注意：这里需要确保模型已经进行过前向传播并收集了注意力权重
    # 我们假设模型的forward方法返回(output, attn_weights)
    
    # 从模型中获取最近一次前向传播的注意力权重
    # 通常在模型评估或训练后调用此函数
    if not hasattr(model, 'last_attn_weights'):
        print("模型没有存储最近的注意力权重，请先运行模型")
        return
    
    attn_weights = model.last_attn_weights
    
    # 差分注意力权重结构通常为: [layers][heads]
    # 我们需要将其转换为与原始plot_attention函数兼容的格式
    
    # 遍历每一层
    for layer_idx, layer_weights in enumerate(attn_weights):
        fig, axs = plt.subplots(1, len(layer_weights), figsize=(5*len(layer_weights), 4))
        
        # 如果只有一个头，确保axs是一个列表
        if len(layer_weights) == 1:
            axs = [axs]
        
        # 遍历每个头的注意力权重
        for head_idx, head_weight in enumerate(layer_weights):
            # 将注意力权重转移到CPU
            head_weight_cpu = head_weight.detach().cpu()
            
            # 检查维度并处理
            if len(head_weight_cpu.shape) == 3:
                print(f"注意力权重形状为 {head_weight_cpu.shape}，取第一个样本进行可视化")
                # 如果是3维张量，取第一个样本 (batch, seq_len, seq_len) -> (seq_len, seq_len)
                head_weight_cpu = head_weight_cpu[0]
            elif len(head_weight_cpu.shape) > 3:
                print(f"注意力权重维度过高: {head_weight_cpu.shape}，取平均值进行可视化")
                # 如果维度更高，取所有批次的平均值
                head_weight_cpu = head_weight_cpu.mean(dim=0)
            
            # 转换为numpy数组
            head_weight_np = head_weight_cpu.numpy()
            
            # 绘制热力图
            heatmap = seaborn.heatmap(head_weight_np, ax=axs[head_idx])
            axs[head_idx].set_title(f"Head {head_idx}", fontsize=10)
            
        # 添加总标题
        plt.suptitle(f"Layer {layer_idx} Differential Attention", fontsize=14)
        plt.tight_layout()
        
        # 保存热力图
        fig.savefig(f'plots/{folder}/diff_attention_layer_{layer_idx}.png')
        plt.clf()
    
    # 创建所有头的平均注意力热力图
    for layer_idx, layer_weights in enumerate(attn_weights):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 处理每个头的权重并计算平均值
        processed_heads = []
        for head_weight in layer_weights:
            head_weight_cpu = head_weight.detach().cpu()
            
            # 检查维度并处理
            if len(head_weight_cpu.shape) == 3:
                # 如果是3维张量，取第一个样本
                head_weight_cpu = head_weight_cpu[0]
            elif len(head_weight_cpu.shape) > 3:
                # 如果维度更高，取所有批次的平均值
                head_weight_cpu = head_weight_cpu.mean(dim=0)
                
            processed_heads.append(head_weight_cpu)
        
        # 计算所有头的平均注意力权重
        all_heads = torch.stack(processed_heads)
        avg_attention = torch.mean(all_heads, dim=0).numpy()
        
        # 绘制平均热力图
        heatmap = seaborn.heatmap(avg_attention, ax=ax)
        ax.set_title(f"Layer {layer_idx} - Average Attention Across All Heads", fontsize=12)
        
        # 保存热力图
        fig.savefig(f'plots/{folder}/diff_attention_layer_{layer_idx}_avg.png')
        plt.clf()
