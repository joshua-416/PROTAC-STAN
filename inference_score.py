import random

import numpy as np
import pandas as pd
import toml
import torch

from data_loader import PROTACLoader
from model import PROTAC_STAN
import argparse

import os.path as osp

def setup_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(model, test_loader, device, save_att=False): 
    model = model.to(device)
    model.eval()

    predictions = []
    scores = []
    att_maps = []
    
    with torch.no_grad():

        for data in test_loader:
            protac_data = data['protac'].to(device)
            e3_ligase_data = data['e3_ligase'].to(device)
            poi_data = data['poi'].to(device)
            # label = data['label'].to(device)

            outputs, atts = model(protac_data, e3_ligase_data, poi_data, mode='eval')
            #_, predicted = torch.max(outputs.data, dim=1)
            #predictions.extend(predicted.cpu().numpy())
            probs = torch.exp(outputs)            # shape: [B, 2]
            pos_scores = probs[:, 1]              # class=1 probability
            scores.extend(pos_scores.cpu().tolist())

            if save_att:
                att_maps.extend(atts.cpu().numpy())

    '''
    results = {
        'predictions': predictions,
        'att_maps': att_maps
    }
    '''    
    results = {'scores': scores, 'att_maps': att_maps}

    return results


def main():

    setup_seed(21332)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    cfg = toml.load('config.toml')
    model_cfg = cfg['model']

    model = PROTAC_STAN(model_cfg)
    path = 'saved_models/protac-stan.pt'
    print(f'Loading model from {path}...')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print(model)
    
    parser = argparse.ArgumentParser(description='PROTAC-STAN Inference')
    parser.add_argument('--root', type=str, default='data/custom', help='Path to the data directory')
    parser.add_argument('--name', type=str, default='custom', help='Raw file name without extension')
    parser.add_argument('--save_att', action='store_true', help='Whether to save attention maps, might consume a lot of memory')

    args = parser.parse_args()

    root = args.root
    name = args.name
    save_att = args.save_att

    _, test_loader = PROTACLoader(root=root, name=name, batch_size=1, train_ratio=0.0)

    results = test(model, test_loader, device, save_att)
    
    #predictions = results['predictions']
    scores = results['scores']
    if save_att:
        att_maps = results['att_maps']
        print('Saving attention maps...')
        np.save(f'{root}/{name}_att.npy', att_maps)
        
    #print(predictions)
    print([round(s, 4) for s in scores])
    csv_path = osp.join(root, f'{name}.csv')          # 예: data/custom/custom.csv

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    if len(df) != len(scores):
        print(f'[WARN] CSV 행수({len(df)})와 예측 개수({len(scores)})가 다릅니다. '
              '일치하는 범위까지만 저장합니다.')
    n = min(len(df), len(scores))
    df.loc[:n-1, 'pred'] = scores[:n]                 # 앞에서부터 차례대로 매핑

    pred_path = osp.abspath(csv_path).split('.')[0]+'_pred.csv'
    df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path} (added column: 'pred')")

if __name__ == '__main__':
    main()
