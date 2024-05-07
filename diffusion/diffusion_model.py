"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

"""
import math
import numpy as np
import torch as th
from tqdm.auto import tqdm
import pdb
from diffusion.nn import mean_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood


""" def normal_kl(mean1, logvar1, mean2, logvar2):
  
  KL divergence between normal distributions parameterized by mean and log-variance.
  
  return 0.5 * (-1.0 + logvar2 - logvar1 + tf.exp(logvar1 - logvar2)
                + tf.squared_difference(mean1, mean2) * tf.exp(-logvar2))
 """

##计算正向过程的β
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "quadratic":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
        )**2
    elif schedule_name == "sigmoid":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        return np.sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



class GaussianDiffusion:
    """
    Contains utilities for the diffusion model.
    Arguments:
    - what the network predicts (x_{t-1}, x_0, or epsilon)
    - which loss function (kl or unweighted MSE)
    - what is the variance of p(x_{t-1}|x_t) (learned, fixed to beta, or fixed to weighted beta)
    - what type of decoder, and how to weight its loss? is its variance learned too?
    """
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type,rescale_timesteps=False):
        self.model_mean_type = model_mean_type  # xprev, xstart, eps
        self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
        self.loss_type = loss_type  # kl, mse
        self.rescale_timesteps = rescale_timesteps

        assert isinstance(betas, np.ndarray)
        self.betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        assert len(betas.shape) == 1, "betas must be 1-D"
        
        #define timesteps's length
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas  #α
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  #Παt
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])  #Πat-1
        assert self.alphas_cumprod_prev.shape == (timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)  #根号Παt
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod) #根号（1-Παt）
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)  #log（-Παt）
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)  #根号（1/Παt）
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1) #根号（1/Παt-1）

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_mean_variance(self, x_start, t):
        """
            Get the distribution q(x_t | x_0).

            :param x_start: the [N x C x ...] tensor of noiseless inputs.
            :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
            :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract_into_tensor(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)* noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool, return_pred_xstart: bool):
        """
            Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
            the initial x, x_0.

            :param model: the model, which takes a signal and a batch of timesteps
                        as input.
            :param x: the [N x C x ...] tensor at time t.
            :param t: a 1-D Tensor of timesteps.
            :param clip_denoised: if True, clip the denoised signal into [-1, 1].
            :param denoised_fn: if not None, a function which applies to the
                x_start prediction before it is used to sample. Applies before
                clip_denoised.
            :param model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            :return: a dict with the following keys:
                    - 'mean': the model mean output.
                    - 'variance': the model variance output.
                    - 'log_variance': the log of 'variance'.
                    - 'pred_xstart': the prediction for x_0.
        """
        B, C, W = x.shape
        # print('B',B)
        # print('t.shape ',t.size())
        #assert t.shape == [B]
        model_output = denoise_fn(x, self._scale_timesteps(t))

        # Learned or fixed variance?
        if self.model_var_type == 'learned':
            # print('model shape',model_output.shape,[B, C * 2, W])
            # assert model_output.shape == [B, C * 2, H, W]
            model_output, model_log_variance = th.split(model_output, C, dim=1)
            model_variance = th.exp(model_log_variance)
        elif self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
            'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
            'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.model_var_type]
            model_variance = self._extract_into_tensor(model_variance, t, x.shape) * th.ones(x.shape)
            model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape) * th.ones(x.shape)
        else:
            raise NotImplementedError(self.model_var_type)

        # Mean parameterization
        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type == 'xstart':  # the model predicts x_0
            pred_xstart = process_xstart(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        elif self.model_mean_type == 'eps':  # the model predicts epsilon
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, pred_xstart
        else:
            return model_mean, model_variance, model_log_variance
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
            assert x_t.shape == eps.shape
            return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
            )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            self._extract_into_tensor(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            self._extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
        
    # === Sampling ===
    def p_sample(self, denoise_fn, *, x, t, clip_denoised=True, return_pred_xstart: bool):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
        denoise_fn, x=x, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        noise = th.randn_like(x)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, denoise_fn, *, shape, noise=None,clip_denoised=True,device=None):
        """
        Generate samples
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            denoise_fn=denoise_fn,
            shape=shape,
            noise=noise,
            device=device,
            clip_denoised=clip_denoised
        ):
            final = sample
        return final[0]

    def p_sample_loop_progressive(self, denoise_fn, *, shape, clip_denoised=True, noise=None, device=None):
        """
        Generate samples and keep track of prediction of x0
        """
        if device is None:
            device = next(denoise_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        
        indices = tqdm(indices)
        #sample xt-1 from xt
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    denoise_fn=denoise_fn,
                    x=img,
                    t=t,
                    clip_denoised=clip_denoised,
                    return_pred_xstart=True
                )
                yield out
                img = out[0]
                
    # === Log likelihood calculation ===

    def _vb_terms_bpd(self, denoise_fn, x_start, x_t, t, *, clip_denoised: bool, return_pred_xstart: bool):
        #得到变分下界
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
        denoise_fn, x=x_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = mean_flat(kl) / np.log(2.)

        decoder_nll = -discretized_gaussian_log_likelihood(
        x_start, means=model_mean, log_scales=0.5 * model_log_variance)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        #assert kl.shape == decoder_nll.shape == t.shape == [x_start.shape[0]]
        output = th.where((t == 0), decoder_nll, kl)
        return (output, pred_xstart) if return_pred_xstart else output

    def training_losses(self, denoise_fn, x_start, t, x_full_start, noise=None):
        """
        Training loss calculation
        """
        # Add noise to data
        # assert t.shape == x_start.shape[0]
     
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape and noise.dtype == x_start.dtype

        loss_fn = {}
        print("train loss x_start",x_start.size())
        print("t",t.size())
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
       
        
        # Calculate the loss
        if self.loss_type == 'kl':  # the variational bound
            loss_fn["loss"] = self._vb_terms_bpd(
                denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, return_pred_xstart=False)
        elif self.loss_type == 'mse':  # unweighted MSE
            #assert self.model_var_type != "learned"
            print('use mse')
            target = {
                'xprev': self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                'xstart': x_start,
                'eps': noise
            }[self.model_mean_type]
            
            print('x_t',x_t.size())
            
            model_output = denoise_fn(x_t, self._scale_timesteps(t))
            if self.model_var_type == "learned":
                print('here')
                B, C = x_t.shape[:2]
                print(model_output.shape)
                print(x_t.shape)
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                loss_fn["vb"] = self._vb_terms_bpd(
                    denoise_fn=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    return_pred_xstart=False
                )
            if self.model_mean_type=='xstart':
                target=x_full_start
                
            assert model_output.shape == target.shape == x_start.shape
            #loss_fn = th.nn.MSELoss(reduction='none')
            loss_fn["mse"] = mean_flat((target - model_output)** 2)
            if "vb" in loss_fn:
                loss_fn["loss"] = loss_fn["mse"] + loss_fn["vb"]
            else:
                loss_fn["loss"] = loss_fn["mse"]
        
        elif self.loss_type == 'ownloss': 
            print('use own loss')
            
            x_t_full = self.q_sample(x_start=x_full_start, t=t, noise=noise)
            
            target = {
                'xprev': self.q_posterior_mean_variance(x_start=x_full_start, x_t=x_t_full, t=t)[0],
                'xstart': x_full_start,
                'eps': noise
            }[self.model_mean_type]
            
            # print('x_t',x_t.size())
            
            model_output = denoise_fn(x_t, self._scale_timesteps(t))
            
            if self.model_var_type == "learned":
                print('here')
                B, C = x_t.shape[:2]
                # print(model_output.shape)
                # print(x_t.shape)
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                loss_fn["vb"] = self._vb_terms_bpd(
                    denoise_fn=lambda *args, r=frozen_out: r,
                    x_start=x_full_start,   #true x_start
                    x_t=x_t,           #pred x_t
                    t=t,
                    clip_denoised=False,
                    return_pred_xstart=False
                )
                
            assert model_output.shape == target.shape == x_start.shape
            #loss_fn = th.nn.MSELoss(reduction='none')
            loss_fn["mse"] = mean_flat((target - model_output)** 2)
            if "vb" in loss_fn:
                loss_fn["loss"] = loss_fn["mse"] + loss_fn["vb"]
            else:
                loss_fn["loss"] = loss_fn["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        # assert losses.shape == t.shape
        return loss_fn

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.)
        assert kl_prior.shape == x_start.shape
        return mean_flat(kl_prior) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, *, clip_denoised=True):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                output,pred_xstart = self._vb_terms_bpd(
                    denoise_fn,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    return_pred_xstart=True
                )
            vb.append(output)
            xstart_mse.append(mean_flat((pred_xstart - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, pred_xstart)
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
