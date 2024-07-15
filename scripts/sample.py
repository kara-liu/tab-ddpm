import torch
import numpy as np
import zero
import os
from tab_ddpm import GaussianMultinomialDiffusion, GaussianDiffusion1D
from tab_ddpm.utils import FoundNANsError
from utils_train import get_model, make_dataset
from lib import round_columns
import lib

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    n_num = -1, 
    n_cat = -1,
    use_g1d_code: bool = False,
    weight_by: str = '',
    wandb_name: str = ''
):
    f = wandb_name
    use_g1d = use_g1d_code 
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val,
        n_num=n_num,
        n_cat=n_cat,
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    print()
    print(f"X_num shape: {D.X_num['train'].shape[1]}" )
    print(f"X_cat shape: {D.X_cat['train'].shape[1]}" )
    print(f"X_num mean: {D.X_num['train'][:,:5].mean(axis=0)}" )
    print(f"X_cat mean: {D.X_cat['train'][:,:5].mean(axis=0)}" )

    num_numerical_features = D.X_num['train'].shape[1] if D.X_num is not None else 0
    if use_g1d: 
        d_in = len(K) + num_numerical_features
    else:
        d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    if use_g1d:
        diffusion = GaussianDiffusion1D(model=model, num_cat_features=len(K),
                    beta_schedule=scheduler, num_numerical_features=num_numerical_features, 
                    timesteps=num_timesteps,
                    objective='pred_x0')
    else: 
        diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            gaussian_loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            scheduler=scheduler,
            device=device
        )

    diffusion.to(device)
    diffusion.eval()
    x_gen = diffusion.sample_all(num_samples, batch_size, y_dist=None, ddim=False)

    # _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    # if disbalance == 'fix':
    #     empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
    #     x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    # elif disbalance == 'fill':
    #     ix_major = empirical_class_dist.argmax().item()
    #     val_major = empirical_class_dist[ix_major].item()
    #     x_gen, y_gen = [], []
    #     for i in range(empirical_class_dist.shape[0]):
    #         if i == ix_major:
    #             continue
    #         distrib = torch.zeros_like(empirical_class_dist)
    #         distrib[i] = 1
    #         num_samples = val_major - empirical_class_dist[i].item()
    #         x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
    #         x_gen.append(x_temp)
    #         y_gen.append(y_temp)
        
    #     x_gen = torch.cat(x_gen, dim=0)
    #     y_gen = torch.cat(y_gen, dim=0)

    # else:
    #     x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    # X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
    X_gen = x_gen.numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###
    
    X_num_ = X_gen
    X_cat = X_num_[:, num_numerical_features:]
    # if num_numerical_features < X_gen.shape[1]:
    #     np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
    #     # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
    #     if T_dict['cat_encoding'] == 'one-hot':
    #         X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
    #     X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])
    # import ipdb; ipdb.set_trace()

    if num_numerical_features != 0:
        # import ipdb; ipdb.set_trace()
        np.save(os.path.join(parent_dir, f'X_num_train_raw{f}'), X_gen[:, :num_numerical_features])
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        # np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        # X_num_ = D.num_transform.invserse_transform(X_gen[:, 6026:num_numerical_features])
        X_num = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        # X_cat_ = X_gen[:, :6026]
        # def round_columns(X):
        #     uniq = np.array([0,1])
        #     for i in range(X.shape[1]):
        #         dist = cdist(X[:, i][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        #         X[:, i] = uniq[dist.argmin(axis=1)]
        #     return X
        # X_cat = round_columns(X_cat_)
        # X_num = np.concatenate([X_cat_, X_num_], axis=1)
        # X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        # disc_cols = []
        # for col in range(X_num_real.shape[1]):
        #     uniq_vals = np.unique(X_num_real[:, col])
        #     if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
        #         disc_cols.append(col)
        # print("Discrete cols:", disc_cols)
        # if model_params['num_classes'] == 0:
        #     y_gen = X_num[:, 0]
        #     X_num = X_num[:, 1:]
        # if len(disc_cols):
        #     X_num = round_columns(X_num_real, X_num, disc_cols)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, f'X_num_train{f}'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir,f'X_cat_train{f}'), X_cat)
    print(f"X_num mean: {X_num[:,:5].mean(axis=0)}" )
    print(f"X_cat mean: {X_cat[:,:5].mean(axis=0)}" )
    # np.save(os.path.join(parent_dir, 'y_train'), y_gen)