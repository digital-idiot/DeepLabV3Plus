import tensorflow as tf
from numpy import nan, isnan
from tensorflow import losses
from tensorflow.math import log, cosh
from tensorflow.math import pow as tf_pow
# from tensorflow import function as tf_function
from tensorflow.python.keras import backend as kb


class DiceLossVariants(losses.Loss):
    
    def __init__(self, *args, **kwargs):
        super_args = dict()
        if 'reduction' in kwargs.keys():
            super_args['reduction'] = kwargs['reduction']
        if 'name' in kwargs.keys():
            super_args['name'] = kwargs['name']
        super().__init__(**super_args)
        if len(args) > 0:
            self.loss_name = args[0]
        elif 'loss_name' in kwargs.keys():
            self.loss_name = kwargs['loss_name']
        else:
            self.loss_name = 'custom'
        
        if 'norm' in kwargs.keys():
            assert isinstance(kwargs['norm'], bool), '`norm`: Illegal type'
            self.norm = kwargs['norm']
        else:
            self.norm = True
        
        if 'eps' in kwargs.keys():
            self.eps = kwargs['eps']
        else:
            self.eps = 1e-8
            
        if 'preserve_axis' in kwargs.keys():
            preserve_axis = kwargs['preserve_axis']
            if isinstance(preserve_axis, int):
                preserve_axis = (preserve_axis,)
            assert isinstance(
                preserve_axis, (tuple, list)
            ), '`preserve_axis`: Illegal type!'
            self.preserve_axis = kwargs['preserve_axis']
        else:
            self.preserve_axis = (0, -1)
        
        if self.loss_name == 'dice':
            self.alpha = 0.5
            self.beta = 0.5
            self.gamma = 1.0
        
        elif self.loss_name == 'iou':
            self.alpha = 1.0
            self.beta = 1.0
            self.gamma = 1.0

        elif self.loss_name == 'tversky':
            self.gamma = 1.0
            if 'alpha' in kwargs.keys():
                self.alpha = kwargs['alpha']
                self.beta = 1 - self.alpha
            elif 'beta' in kwargs.keys():
                self.beta = kwargs['beta']
                self.alpha = 1 - self.beta
            else:
                self.alpha = 0.3
                self.beta = 0.7

        elif self.loss_name == 'focal_tversky':
            if 'alpha' in kwargs.keys():
                self.alpha = kwargs['alpha']
                self.beta = 1 - self.alpha
            elif 'beta' in kwargs.keys():
                self.beta = kwargs['beta']
                self.alpha = 1 - self.beta
            else:
                self.alpha = 0.3
                self.beta = 0.7
            
            if 'gamma' in kwargs.keys():
                self.gamma = kwargs['gamma']
            else:
                self.gamma = 4.0 / 3.0

        elif self.loss_name == 'log-cosh':
            self.gamma = nan
            if 'alpha' in kwargs.keys():
                self.alpha = kwargs['alpha']
                self.beta = 1 - self.alpha
            elif 'beta' in kwargs.keys():
                self.beta = kwargs['beta']
                self.alpha = 1 - self.beta
            else:
                self.alpha = 0.5
                self.beta = 0.5

        elif self.loss_name == 'custom':
            if 'alpha' in kwargs.keys():
                self.alpha = kwargs['alpha']
                self.beta = 1 - self.alpha
            elif 'beta' in kwargs.keys():
                self.beta = kwargs['beta']
                self.alpha = 1 - self.beta
            else:
                self.alpha = 0.5
                self.beta = 0.5
            
            if 'gamma' in kwargs.keys():
                self.gamma = kwargs['gamma']
            else:
                self.gamma = 1.0
        else:
            raise NotImplementedError(
                "Unknown Function Name: {}".format(self.loss_name)
            )
        variables = (self.alpha, self.beta, self.gamma, self.eps)
        vnames = ('alpha', 'beta', 'gamma', 'eps')
        for v, n in zip(variables, vnames):
            assert isinstance(
                v, (int, float)
            ), "`{}`: '{}' is illegal value".format(n, v)

    def call(self, y_true, y_pred):
        
        alpha = self.alpha
        beta = self.beta
        eps = self.eps
        preserve_axis = self.preserve_axis

        # once = kb.ones(kb.shape(y_true))
        once = tf.ones_like(y_true)
        p0 = y_pred  # probability that voxels are class i
        p1 = once - y_pred  # probability that voxels are not class i
        g0 = y_true
        g1 = once - y_true

        assert all(
            [
                (
                    isinstance(n, int) and (
                        (0 <= n < kb.ndim(p0)) or (-kb.ndim(p0) <= n < 0)
                    )
                )
                for n in preserve_axis
            ]
        ), "`preserve_axis`: Illegal Value"

        dims = list(range(kb.ndim(p0)))
        preserve_axis = list(set(preserve_axis))
        for ax in preserve_axis:
            del dims[ax]

        numerator = kb.sum(
            x=p0 * g0,
            axis=dims
        ) + eps
        denominator = numerator + alpha * kb.sum(
            x=p0 * g1,
            axis=dims
        ) + beta * kb.sum(
            x=p1 * g0,
            axis=dims
        ) + eps

        t_values = numerator / denominator

        loss_tensor = (
            kb.ones_like(
                x=t_values, dtype=t_values.dtype
            ) - t_values
        )
        if isnan(self.gamma):
            loss_tensor = log(cosh(loss_tensor))
        else:
            if self.gamma != 1:
                loss_tensor = tf_pow(loss_tensor, self.gamma)

        if self.norm:
            return kb.mean(x=loss_tensor, axis=None, keepdims=False)
        else:
            return kb.sum(x=loss_tensor, axis=None, keepdims=False)
