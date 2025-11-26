import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

# 定义color类用于控制终端输出颜色
class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]  # cut
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])  # pad
        windows.append(w if 'DTAAD' in model.name or 'SDfomer' in model.name or 'TranAD' in model.name else w.view(-1))
    return torch.stack(windows)


def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    # loader = [i[:, debug:debug+1] for i in loader]
    if args.less: loader[0] = cut_array(0.25, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    # 根据是否使用了less参数来确定保存路径
    if args.less:
        folder = f'checkpoints/{args.model}_{args.dataset}_less/'
    else:
        folder = f'checkpoints/{args.model}_{args.dataset}/'
    
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    
    # 根据是否使用了less参数来确定加载路径
    if args.less:
        fname = f'checkpoints/{args.model}_{args.dataset}_less/model.ckpt'
    else:
        fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        l2s = []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s = []
            for d in data:
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            model.to(torch.device(args.Device))
            hidden = None  # 显式初始化hidden为None
            for i, d in enumerate(data):
                # 确保每次迭代都使用新的优化器状态
                optimizer.zero_grad()
                
                # 将数据移动到设备上
                d = d.to(torch.device(args.Device))
                
                # 前向传播
                y_pred, mu, logvar, hidden = model(d, hidden)
                
                # 计算损失
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                
                # 记录损失值
                mses.append(torch.mean(MSE).item())
                klds.append(model.beta * torch.mean(KLD).item())
                
                # 反向传播
                loss.backward(retain_graph=True)
                
                # 更新参数
                optimizer.step()
                
                # 分离hidden状态，避免梯度累积
                hidden = hidden.detach()
                
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item());
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        if training:
            # 为MTAD_GAT模型使用批处理
            if 'MTAD_GAT' in model.name:
                # 创建数据加载器，使用较小的批量
                batch_size = 64  # 可以根据内存情况调整
                data_tensor = torch.DoubleTensor(data)
                dataset = TensorDataset(data_tensor, data_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                # 批量训练
                for batch_idx, (batch_data, _) in enumerate(dataloader):
                    batch_data = batch_data.to(torch.device(args.Device))
                    batch_losses = []
                    h = None  # 每个批次开始时重置隐藏状态
                    
                    # 处理批次中的每个样本
                    for i, d in enumerate(batch_data):
                        # 分离隐藏状态，避免梯度累积
                        if h is not None:
                            h = h.detach()
                        
                        # 前向传播
                        x, h = model(d, h)
                        
                        # 确保x和d的形状匹配
                        if x.shape != d.shape:
                            if len(x.shape) == 1 and len(d.shape) == 2:
                                d_resized = d.view(-1)[:x.shape[0]]
                                loss = l(x, d_resized)
                            else:
                                x_resized = x[:d.shape[0]] if x.shape[0] > d.shape[0] else torch.cat([x, torch.zeros(d.shape[0] - x.shape[0], device=x.device)])
                                loss = l(x_resized, d.view(-1))
                        else:
                            loss = l(x, d)
                        
                        batch_losses.append(torch.mean(loss))
                    
                    # 计算批次的平均损失
                    if batch_losses:
                        batch_loss = torch.mean(torch.stack(batch_losses))
                        l1s.append(batch_loss.item())
                        
                        # 反向传播和优化
                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()
                        
                        # 清理内存
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                # 其他模型保持原有逻辑
                for i, d in enumerate(data):
                    d = d.to(torch.device(args.Device))
                    x = model(d)
                    loss = torch.mean(l(x, d))
                    l1s.append(torch.mean(loss).item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # 更新学习率
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            # 对于 MSCRED、GDN 和 CAE_M 模型，使用混合 CPU/GPU 方法并批处理
            if 'MSCRED' in model.name or 'GDN' in model.name or 'CAE_M' in model.name:
                # 确保模型仍在GPU上
                device = torch.device(args.Device)
                model.to(device)
                
                # 使用较小的批量进行处理，避免 GPU 内存不足
                batch_size = 64  # 可以根据您的 GPU 内存调整此值
                
                all_losses = []
                all_preds = []
                
                # 将数据分成较小的批次处理
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size]
                    batch = batch.to(device)
                    
                    with torch.no_grad():  # 进一步减少内存使用
                        batch_xs = []
                        for d in batch:
                            if 'MTAD_GAT' in model.name:
                                x, _ = model(d, None)
                            else:
                                x = model(d)
                            batch_xs.append(x)
                        
                        # 立即处理这一批次的结果
                        batch_xs = torch.stack(batch_xs)
                        batch_y_pred = batch_xs[:, batch.shape[1] - feats:batch.shape[1]].view(-1, feats)
                        batch_loss = l(batch_xs, batch)
                        batch_loss = batch_loss[:, batch.shape[1] - feats:batch.shape[1]].view(-1, feats)
                        
                        # 移到 CPU 并立即释放 GPU 内存
                        all_losses.append(batch_loss.cpu().detach().numpy())
                        all_preds.append(batch_y_pred.cpu().detach().numpy())
                    
                    # 手动触发垃圾回收以释放内存
                    torch.cuda.empty_cache()
                
                # 合并所有批次的结果
                loss_np = np.concatenate(all_losses, axis=0)
                y_pred_np = np.concatenate(all_preds, axis=0)
                
                return loss_np, y_pred_np
            elif 'MTAD_GAT' in model.name:
                # 特别处理MTAD_GAT模型，使用批处理
                # 创建数据加载器，使用较小的批量
                batch_size = 64  # 可以根据内存情况调整
                data_tensor = torch.DoubleTensor(data)
                dataset = TensorDataset(data_tensor, data_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                all_xs = []
                
                # 使用设备
                device = torch.device(args.Device if torch.cuda.is_available() else 'cpu')
                model.to(device)
                
                for batch_idx, (batch_data, _) in enumerate(dataloader):
                    batch_data = batch_data.to(device)
                    batch_xs = []
                    h = None  # 每个批次开始时重置隐藏状态
                    
                    with torch.no_grad():  # 不计算梯度，减少内存使用
                        for i, d in enumerate(batch_data):
                            # 前向传播
                            x, h = model(d, h)
                            
                            # 确保x的大小与d匹配
                            if x.shape != d.view(-1).shape:
                                if x.shape[0] < d.view(-1).shape[0]:
                                    # 如果x太小，用零填充
                                    x_padded = torch.zeros(d.view(-1).shape[0], device=device, dtype=x.dtype)
                                    x_padded[:x.shape[0]] = x
                                    batch_xs.append(x_padded)
                                else:
                                    # 如果x太大，截断它
                                    batch_xs.append(x[:d.view(-1).shape[0]])
                            else:
                                batch_xs.append(x)
                    
                    # 收集这个批次的结果
                    all_xs.extend(batch_xs)
                    
                    # 清理内存
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 转换为张量
                xs = torch.stack(all_xs)
                
                # 计算预测
                y_pred = xs[:, :feats].view(-1, feats)
                
                # 计算每个样本的损失
                losses = []
                
                # 重新创建数据加载器以计算损失
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                for batch_idx, (batch_data, _) in enumerate(dataloader):
                    batch_data = batch_data.to(device)
                    batch_start = batch_idx * batch_size
                    batch_end = min((batch_idx + 1) * batch_size, len(xs))
                    batch_xs = xs[batch_start:batch_end]
                    
                    with torch.no_grad():
                        for i, (x, d) in enumerate(zip(batch_xs, batch_data)):
                            # 调整大小以匹配
                            d_flat = d.view(-1)
                            x_match = x[:d_flat.shape[0]] if x.shape[0] > d_flat.shape[0] else torch.cat([x, torch.zeros(d_flat.shape[0] - x.shape[0], device=device, dtype=x.dtype)])
                            loss = l(x_match, d_flat)
                            losses.append(loss)
                
                # 转换为numpy数组
                loss = torch.stack(losses).cpu().detach().numpy()
                y_pred = y_pred.cpu().detach().numpy()
                
                return loss, y_pred
            else:
                # 其他模型保持原有逻辑
                model.to(torch.device('cpu'))
                xs = []
                for d in data:
                    if 'MTAD_GAT' in model.name:
                        x, h = model(d, None)
                    else:
                        x = model(d)
                    xs.append(x)
                xs = torch.stack(xs)
                y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                loss = l(xs, data)
                loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        bcel = nn.BCELoss(reduction='mean')
        msel = nn.MSELoss(reduction='mean')

        # 确保标签和模型在同一设备上
        device = torch.device(args.Device)
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])  # label smoothing
        real_label = real_label.type(torch.DoubleTensor).to(device)
        fake_label = fake_label.type(torch.DoubleTensor).to(device)
        
        n = epoch + 1
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                d = d.to(device)
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            # 在GPU上进行测试以降低CPU使用率
            device = next(model.parameters()).device
            data = data.to(device)
            outputs = []
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], [] 
        if training:
            for d, _ in dataloader:  
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0] 
                window = d.permute(1, 0, 2)  
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)  
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'DTAAD' in model.name and 'SDfomer' not in model.name:
        l = nn.MSELoss(reduction='none')
        _lambda = 0.8
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        
        # 修改这里：测试时也使用批量处理
        # 对于Mamba模型，使用较小的批量以避免内存不足
        if hasattr(model, 'using_official_mamba') and model.using_official_mamba and not training:
            # 对于Mamba模型，测试时使用与训练相同的批量大小
            bs = model.batch
        else:
            # 对于其他模型，保持原来的逻辑
            bs = model.batch if training else len(data)
            
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                l1 = _lambda * l(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * l(z[1].permute(1, 0, 2), elem)
                
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            # 检查是否为Mamba模型且需要在GPU上运行
            is_mamba_model = hasattr(model, 'using_official_mamba') and model.using_official_mamba
            
            # 如果是Mamba模型且需要保持在GPU上，则不移动到CPU
            if not is_mamba_model:
                model.to(torch.device('cpu'))
            
            # 收集所有批次的预测结果和损失
            all_losses = []
            all_preds = []
            
            # 分批处理测试数据
            for d, _ in dataloader:
                # 对于Mamba模型，确保数据在GPU上
                if is_mamba_model and torch.cuda.is_available():
                    window = d.permute(0, 2, 1).cuda()
                else:
                    window = d.permute(0, 2, 1)
                
                local_bs = d.shape[0]
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                
                # 原始DTAAD模型的损失计算
                z_out = z[1].permute(1, 0, 2)
                batch_loss = l(z_out, elem)[0]
                
                # 将结果移到CPU并转换为numpy
                if z_out.device.type == 'cuda':
                    batch_loss = batch_loss.cpu().detach().numpy()
                    batch_pred = z_out.cpu().detach().numpy()[0]
                else:
                    batch_loss = batch_loss.detach().numpy()
                    batch_pred = z_out.detach().numpy()[0]
                
                # 收集结果
                all_losses.append(batch_loss)
                all_preds.append(batch_pred)
            
            # 合并所有批次的结果
            loss_np = np.concatenate(all_losses, axis=0)
            z_np = np.concatenate(all_preds, axis=0)
            
            return loss_np, z_np
    elif 'SDfomer' in model.name:
        l = nn.MSELoss(reduction='none')
        # 注意力权重正则化系数
        attn_weight_lambda = 0.1
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        # 修改这里：测试时使用较小的批量大小以避免内存不足
        bs = model.batch if training else min(128, len(data))  # 测试时最多使用64个样本
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, attn_losses = [], []
        
        if training:
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                
                # 获取模型输出和注意力权重
                output, attn_weights = model(window)
                output_permuted = output.permute(1, 0, 2)
                
                # 计算重建损失
                recon_loss = l(output_permuted, elem)
                
                # 计算注意力权重正则化损失
                attn_penalty = 0.0
                for layer_weights in attn_weights:
                    for head_weights in layer_weights:
                        # 计算注意力权重的熵，鼓励稀疏性
                        # 首先确保权重为正且和为1
                        head_weights = torch.abs(head_weights)
                        head_sum = head_weights.sum(dim=-1, keepdim=True)
                        head_weights = head_weights / (head_sum + 1e-10)
                        
                        # 计算熵: -sum(p*log(p))
                        eps = 1e-10  # 防止log(0)
                        entropy = -torch.sum(head_weights * torch.log(head_weights + eps), dim=-1)
                        
                        # 我们希望最小化熵，即鼓励注意力集中在少数重要特征上
                        attn_penalty += torch.mean(entropy)
                
                # 总损失 = 重建损失 + 注意力权重正则化
                loss = torch.mean(recon_loss) + attn_weight_lambda * attn_penalty
                #loss = torch.mean(recon_loss)
                l1s.append(torch.mean(recon_loss).item())
                attn_losses.append(attn_penalty.item())
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tRecon Loss = {np.mean(l1s)},\tAttn Loss = {np.mean(attn_losses)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            # 测试阶段 - 使用较小的批量大小并分批处理
            model.to(torch.device('cpu'))
            all_losses = []
            all_preds = []
            
            # 分批处理测试数据以避免内存不足
            for d, _ in dataloader:
                window = d.permute(0, 2, 1)
                local_bs = d.shape[0]
                elem = window[:, :, -1].view(1, local_bs, feats)
                
                # 获取模型输出和注意力权重
                output, attn_weights = model(window)
                output_permuted = output.permute(1, 0, 2)
                
                # 计算重建损失
                batch_loss = l(output_permuted, elem)[0]
                
                # 将结果转换为numpy
                batch_loss = batch_loss.detach().numpy()
                batch_pred = output_permuted.detach().numpy()[0]
                
                # 收集结果
                all_losses.append(batch_loss)
                all_preds.append(batch_pred)
                
                # 手动清理内存
                del output, attn_weights, output_permuted, batch_loss, batch_pred
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 合并所有批次的结果
            loss_np = np.concatenate(all_losses, axis=0)
            z_np = np.concatenate(all_preds, axis=0)
            
            return loss_np, z_np
    else:
        model.to(torch.device(args.Device))
        data = data.to(torch.device(args.Device))
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    # 添加异常检测以帮助诊断梯度问题
    if args.model == 'OmniAnomaly':
        torch.autograd.set_detect_anomaly(True)

    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
                      'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name or 'SDfomer' in model.name:
        # 所有模型都使用窗口转换
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs =5
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        
        # 根据是否使用了less参数生成不同的文件名
        if args.less:
            plot_name = f'{args.model}_{args.dataset}_less'
        else:
            plot_name = f'{args.model}_{args.dataset}'
            
        plot_accuracies(accuracy_list, plot_name)

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    start_inference = time()
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    print(color.BOLD + 'Inference time: ' + "{:10.4f}".format(time() - start_inference) + ' s' + color.ENDC)

    ### Plot curves
    if not args.test:
        if 'TranAD' in model.name or 'DTAAD' in model.name or 'SDfomer' in model.name: testO = torch.roll(testO, 1, 0)
        
        # 根据是否使用了less参数生成不同的文件名
        if args.less:
            plot_name = f'{args.model}_{args.dataset}_less'
        else:
            plot_name = f'{args.model}_{args.dataset}'
            
        plotter(plot_name, testO, y_pred, loss, labels)

    ### Plot attention
    if not args.test:
        if 'DTAAD' in model.name or 'SDfomer' in model.name:
            # 添加检查模型是否禁用了注意力可视化
            if not hasattr(model, 'no_attention_visualization') or not model.no_attention_visualization:
                # 根据是否使用了less参数生成不同的文件名
                if args.less:
                    plot_name = f'{args.model}_{args.dataset}_less'
                else:
                    plot_name = f'{args.model}_{args.dataset}'
                    
                plot_attention(model, 1, plot_name)
                
                # 为差分注意力模型添加专门的可视化
                if 'Diff' in model.name or 'SDfomer' in model.name:
                    from src.utils import plot_differential_attention
                    # 根据是否使用了less参数生成不同的文件名
                    if args.less:
                        diff_plot_name = f'{args.model}_{args.dataset}_less_diff'
                    else:
                        diff_plot_name = f'{args.model}_{args.dataset}_diff'
                        
                    plot_differential_attention(model, diff_plot_name)
            else:
                print(f"模型 {args.model} 已禁用注意力热力图可视化")

    ### Scores
    df = pd.DataFrame()
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    # 确保不会超出边界
    n_dims = min(loss.shape[1], lossT.shape[1], labels.shape[1])
    for i in range(n_dims):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df = df._append(result, ignore_index=True)
    # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
    # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(df)
    pprint(result)
    # pprint(getresults2(df, result))
