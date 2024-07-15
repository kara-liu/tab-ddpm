from copy import deepcopy
import torch
import os
import numpy as np
import zero
from tab_ddpm import GaussianMultinomialDiffusion, GaussianDiffusion1D, Trainer1D
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd
import wandb 
from tqdm import tqdm 
class Trainer:
    def __init__(self, diffusion, train_loader, lr, weight_decay, steps, train_iters:int, val_iters:int,
                        dataset, device=torch.device('cuda:1'), val_loader=None,
                        tr_iter_per_epoch:int=None,):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        # self.inverse_transform = dataset.num_transform.inverse_transform
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.train_iters = train_iters
        self.eval_every_nth_epoch = 5
        self.sample_every_nth_epoch = 20
        self.val_iters = val_iters
        self.steps = steps 

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        # for k in out_dict:
        #     out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)

        
        loss = loss_multi + loss_gauss
        # clip_value = 2
        # for p in self.diffusion.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 1)

        if torch.isnan(loss):
            import ipdb; ipdb.set_trace()
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def _run_step_val(self, x, out_dict):
        x = x.to(self.device)
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        return loss_multi, loss_gauss

    def run_loop(self):
        # step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        # while step < self.steps:
        for step in tqdm(range(self.steps)):
            x = next(self.train_loader)
            assert len(x)==1
            x = x[0]
            out_dict = {'y': None}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            try:
                curr_loss_multi += batch_loss_multi.item() * len(x)
            except:
                curr_loss_multi += 0
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0
                wandb.log({
                    "iteration": step,
                    "gloss": gloss,
                    "mloss": mloss,
                    "loss": mloss+gloss
                })
            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())


            if (step + 1) % (self.sample_every_nth_epoch * self.train_iters) == 0:
                self.diffusion.eval()
                X_gen = self.diffusion.sample_all(2048, 1024, y_dist=None, ddim=False).numpy()
                X_num = X_gen[:, :self.diffusion.num_numerical_features:]
                X_cat = X_gen[:, self.diffusion.num_numerical_features:]
                # X_num = self.inverse_transform(X_num)
                print(f"Cat mean at step {step}: {X_cat[:,:5].mean(axis=0)}")
                print(f"Num mean at step {step}: {X_num[:,:5].mean(axis=0)}")
                # print(f"Num unique at step {step}: {np.unique(X_num[:,0])}")
                self.diffusion.train()
            if (step + 1) % (self.eval_every_nth_epoch * self.train_iters) == 0:
                self.diffusion.eval()
                curr_count_val = 0
                curr_loss_gauss_val = 0.0
                curr_loss_multi_val = 0.0
                with torch.no_grad():
                    for _ in range(self.val_iters):
                        x = next(self.val_loader)
                        assert len(x)==1
                        x = x[0]
                        batch_loss_multi, batch_loss_gauss = self._run_step_val(x, out_dict={'y': None})

                        curr_count_val += len(x)
                        try:
                            curr_loss_multi_val += batch_loss_multi.item() * len(x)
                        except:
                            curr_loss_multi_val += 0
                        curr_loss_gauss_val += batch_loss_gauss.item() * len(x)

                    gloss = np.around(curr_loss_gauss_val / curr_count_val, 4)
                    mloss = np.around(curr_loss_multi_val / curr_count_val, 4)
                    print(f'Validaiton/MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')

                wandb.log({
                    "val/gloss": gloss,
                    "val/mloss": mloss,
                    "val/loss": mloss+gloss
                })
                self.diffusion.train()

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    n_num = -1, 
    n_cat = -1,
    use_g1d_code: bool = False,
    weight_by: str = '',
    wandb_name: str = ''
):

    use_g1d = use_g1d_code 
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val,
        n_num=n_num,
        n_cat=n_cat,
        weight_by=weight_by
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    
    print()
    print(f"X_num shape: {dataset.X_num['train'].shape[1]}" )
    print(f"X_cat shape: {dataset.X_cat['train'].shape[1]}" )
    print(f"X_num mean: {dataset.X_num['train'][:,:5].mean(axis=0)}" )
    print(f"X_cat mean: {dataset.X_cat['train'][:,:5].mean(axis=0)}" )
    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0

    if use_g1d: 
        d_in = len(K) + num_numerical_features
    else:
        d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in

    print(model_params)

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    val_loader = lib.prepare_fast_dataloader(dataset, split='val', batch_size=batch_size)
    train_iters = int(dataset.X_num['train'].shape[0]// batch_size) + 1
    val_iters = int(dataset.X_num['val'].shape[0]// batch_size) + 1

    if use_g1d:
        diffusion = GaussianDiffusion1D(model=model, num_cat_features=len(K),
                    beta_schedule=scheduler, num_numerical_features=num_numerical_features, 
                    timesteps=num_timesteps,
                    objective='pred_x0')
        diffusion.to(device)
        diffusion.eval()
        if weight_by == '':
            print("Setting source as .X_num_targ_mean!")
            X_num_targ_mean = np.load('/oak/stanford/groups/rbaltman/karaliu/tab-ddpm/data/Xs_num_mean.npy')
            X_num_targ_max = np.load('/oak/stanford/groups/rbaltman/karaliu/tab-ddpm/data/Xs_num_max.npy')
            X_cat_targ_mean = np.load('/oak/stanford/groups/rbaltman/karaliu/tab-ddpm/data/Xs_cat_mean.npy')
        else:
            print("Setting target as .X_num_targ_mean!")
            X_num_targ_mean = np.load('/oak/stanford/groups/rbaltman/karaliu/tab-ddpm/data/Xt_num_mean.npy')
            X_num_targ_max = np.load('/oak/stanford/groups/rbaltman/karaliu/tab-ddpm/data/Xt_num_max.npy')
            X_cat_targ_mean = np.load('/oak/stanford/groups/rbaltman/karaliu/tab-ddpm/data/Xt_cat_mean.npy')
        
        trainer = Trainer1D(diffusion_model=diffusion, tr_loader=train_loader, train_batch_size=batch_size,
            train_num_steps=steps, X_cat_targ_mean=X_cat_targ_mean,
            X_num_targ_mean_max=(X_num_targ_mean, X_num_targ_max),
            num_transform=dataset.num_transform.inverse_transform)
        trainer.train()
        wandb.finish()
        torch.save(diffusion.model.state_dict(), os.path.join(parent_dir, f'model{wandb_name}.pt'))
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
        diffusion.train()
        trainer = Trainer(
            diffusion,
            train_loader,
            lr=lr,
            steps=steps,
            weight_decay=weight_decay,
            train_iters=train_iters, val_iters=val_iters,
            device=device,
            val_loader=val_loader,
            dataset=dataset
        )
        trainer.run_loop()
        wandb.finish()
        # trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
        torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, f'model{weight_by}.pt'))
        torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))



    