import tensorflow as tf
import numpy as np
import time


class Settings:
    def __init__(self, state_dim, action_dim, writer_f):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.writer_f = writer_f


class A2C2_d:
    def __init__(self, settings):
        self.sess = tf.Session()

        assert isinstance(settings, Settings)
        self.state_dim = settings.state_dim
        self.action_dim = settings.action_dim
        self.writer = tf.summary.FileWriter('./logs/A2C2-carpole/%s/' % str(time.time()), self.sess.graph) if settings.writer_f else None

        self.initializer = tf.random_normal_initializer(0., 0.1)

        self.state_layer = tf.placeholder(tf.float32, [None, self.state_dim], name='state')

        pi, value, vars = self._build_net('net', self.state_layer, trainable=False)
        pi_, value_, vars_ = self._build_net('net_', self.state_layer, trainable=True)

        self.update_op = [var.assign(var_) for var, var_ in zip(vars, vars_)]

        self.forward_op = [pi, tf.squeeze(value, axis=0)]

        # self.kl = tf.distributions.kl_divergence(pi, pi_)


        with tf.variable_scope('training'):
            self.action_fb = tf.placeholder(tf.float32, [None, self.action_dim], 'action_fb')
            self.return_fb = tf.placeholder(tf.float32, [None, 2], 'return_fb')
            # self.advantage_fb = tf.placeholder(tf.float32, [None, 1], 'advantage_fb')
            self.advantage_fb =self.return_fb - tf.stop_gradient(value_)
            advantage_sum = tf.reduce_sum(self.advantage_fb, axis=1)

            ratio = tf.reduce_sum(pi_*self.action_fb, axis=1) / tf.reduce_sum(pi*self.action_fb + 1e-8, axis=1)
            loss = tf.multiply(ratio, advantage_sum)
            clipped_loss = tf.multiply(tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2), advantage_sum)
            self.actor_loss = -tf.reduce_mean(tf.minimum(loss, clipped_loss))

            # self.actor_loss = -tf.reduce_mean(tf.log(tf.reduce_sum(self.action_fb * pi_, reduction_indices=1)+1e-8) * self.advantage_fb)

            self.critic_loss = tf.losses.mean_squared_error(self.return_fb, value_)

            self.actor_train_op = tf.train.AdamOptimizer(0.002).minimize(self.actor_loss)
            self.critic_train_op = tf.train.AdamOptimizer(0.001).minimize(self.critic_loss)
            self.train_op = [self.actor_train_op, self.critic_train_op]

        with tf.name_scope('summary'):
            tf.summary.scalar('actor_loss', self.actor_loss)
            tf.summary.scalar('critic_loss', self.critic_loss)
            self.merged = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.next_state = None
        self.eps = 0

    def forward(self, state):
        s = state
        if len(s.shape) == 1:
            s = [s]
        feed_dict = {self.state_layer: s}
        pi, value = self.sess.run(self.forward_op, feed_dict=feed_dict)
        a = np.random.choice(self.action_dim, p=np.squeeze(pi))
        return a, value

    def buffer_append(self, state, action, reward, value, next_state):
        self.states.append(state)
        self.actions.append(np.eye(self.action_dim)[action])
        self.rewards.append(reward)
        self.values.append(value)
        self.next_state = next_state

    def buffer_clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.next_state = None

    def n_step_return(self, done, gamma=0.95, n_step=32):
        r = np.array(self.rewards)
        batch_size = r.shape[0]
        obj_num = r.shape[1]
        gamma = gamma * np.array([1]*obj_num)

        _, R = self.forward(self.next_state)
        if done:
            R[0] = 0. - 0.99

        returns = np.zeros_like(r)

        multiplier = np.array([gamma**n for n in range(n_step)])
        for t in reversed(range(batch_size)):
            if t > batch_size - n_step - 1:
                R = r[t] + gamma * R
                returns[t] = R
            else:
                returns[t] = np.sum(multiplier * r[t:t + n_step]) + self.values[t + n_step]
        return returns

    def lambda_return(self, done, gamma=0.95, lam=0.9):

        r = np.array(self.rewards)
        batch_size = r.shape[0]
        obj_num = r.shape[1]

        _, R = self.forward(self.next_state)
        if done:
            R[0] = 0. - 0.99
        self.values.append(R)
        vs = np.array(self.values)

        returns = np.zeros_like(r)

        for t in reversed(range(batch_size)):
            R = r[t] + (1-lam) * gamma * vs[t+1] +\
                lam * gamma * R
            returns[t] = R

        return returns

    def train(self, done, return_method=None):
        if return_method is None:
            return_method = self.lambda_return
        returns = return_method(done)

        feed_dict = {self.state_layer: np.array(self.states),
                     self.action_fb: np.array(self.actions),
                     self.return_fb: np.array(returns)}
        fetch = self.train_op.copy()
        fetch.append(self.merged)
        fetch.append(self.critic_loss)
        for i in range(1):
            _, _, merged, closs = self.sess.run(fetch, feed_dict=feed_dict)
        if self.writer is not None and self.eps % 100 == 0:
            self.writer.add_summary(merged, self.eps/100)
        self.eps += 1

        self.sess.run(self.update_op)

        self.buffer_clear()

        return closs


    def _build_net(self, name, input, trainable):
        with tf.variable_scope(name, initializer=self.initializer):
            layer = tf.layers.dense(input, 32, activation=tf.nn.leaky_relu, trainable=trainable)
            # layer = tf.layers.dense(layer, 16, activation=tf.nn.leaky_relu, trainable=trainable)

            action_ps = tf.layers.dense(layer, self.action_dim, activation=tf.nn.softmax, trainable=trainable)
            # dist = tf.distributions.Categorical(probs=action_ps)

            layer = tf.layers.dense(input, 32, activation=tf.nn.leaky_relu, trainable=trainable)
            layer = tf.layers.dense(layer, 16, activation=tf.nn.leaky_relu, trainable=trainable)
            value_org = tf.layers.dense(layer, 1, trainable=trainable)
            layer = tf.layers.dense(input, 32, activation=tf.nn.leaky_relu, trainable=trainable)
            layer = tf.layers.dense(layer, 16, activation=tf.nn.leaky_relu, trainable=trainable)
            value_pos = tf.layers.dense(layer, 1, trainable=trainable)
            value = tf.concat([value_org, value_pos], axis=1, name='value')

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)

            return action_ps, value, vars
