import numpy as np
import os
import shutil
import tensorflow as tf
import utils

from model import CAModel


class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)



class Experiment:
    """
    Help with running controlled experiments
    """
    def __init__(self,batch_size = 8,cell_fire_rate = 0.5, channel_n = 16,experiment_type="growing",
                 num_epochs=8000, pool_size = 1024, target_emoji = "🐒", target_size = 40, target_padding =16,
                 use_pattern_pool=True,damage_n=0):

        # PRIVATE
        self._batch_size = batch_size
        self._cell_fire_rate = cell_fire_rate # Used during stochastic uddate
        self._channel_n = channel_n
        self._damage_n = damage_n # The numer of patterns to damage for an experiment
        self._experiment_type = experiment_type # growing, persistent, regenerating
        self._num_epochs = num_epochs
        self._pool_size = pool_size
        self._target_emoji = target_emoji
        self._target_padding = target_padding
        self._targent_size = target_size # Size of the target image
        self._use_pattern_pool = use_pattern_pool

        # PUBLIC
        self.model = CAModel()
        self.pad_target_image = None
        self.pool = SamplePool()
        self.target_image = utils.load_emoji(target_emoji)
        lr = 2e-3
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000], [lr, lr * 0.1])
        self.trainer = tf.keras.optimizers.Adam(lr_sched)

        if os.path.isdir("train_log"):
            shutil.rmtree("train_log")
        os.mkdir("train_log")

    def loss_f(self,x,target):
        return tf.reduce_mean(tf.square(utils.to_rgba(x) - target), [-2, -3, -1])

    @tf.function
    def train_step(self,x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = self.model(x)
            loss = tf.reduce_mean(self.loss_f(x,self.pad_target_image))
        grads = g.gradient(loss, self.model.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.model.weights))
        return x, loss

    def run(self):

        loss_log = []
        p = self._target_padding
        pad_target = tf.pad(self.target_image, [(p, p), (p, p), (0, 0)])
        self.pad_target_image = pad_target
        h, w = pad_target.shape[:2]
        seed = np.zeros([h, w, self._channel_n], np.float32)
        seed[h // 2, w // 2, 3:] = 1.0
        self.pool = SamplePool(x=np.repeat(seed[None, ...], self._pool_size, 0))

        for _ in range(self._num_epochs):

            if self._use_pattern_pool:
                batch = self.pool.sample(self._batch_size)
                x0 = batch.x
                loss_rank = self.loss_f(x0,self.pad_target_image).numpy().argsort()[::-1]
                x0 = x0[loss_rank]
                x0[:1] = seed
                if self._damage_n:
                    damage = 1.0 - utils.make_circle_masks(self._damage_n, h, w).numpy()[..., None]
                    x0[-self._damage_n:] *= damage
            else:
                x0 = np.repeat(seed[None, ...], self._batch_size, 0)

            x, loss = self.train_step(x0)

            if self._use_pattern_pool:
                batch.x[:] = x
                batch.commit()

            step_i = len(loss_log)
            loss_log.append(loss.numpy())

            if step_i % 10 == 0:
                utils.generate_pool_figures(self.pool, step_i)
            if step_i % 100 == 0:
                utils.visualize_batch(x0, x, step_i)
                utils.plot_loss(loss_log,step_i)
                utils.export_model(self.model, 'train_log/%04d' % step_i)

            print('\r step: %d, log10(loss): %.3f' % (len(loss_log), np.log10(loss)), end='')

