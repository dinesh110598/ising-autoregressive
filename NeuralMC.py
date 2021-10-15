import tensorflow as tf
from tensorflow import math as tfm
import ising


@tf.function
def neural_mc(model, params, batch_size, beta, n, g=tf.random.Generator.from_non_deterministic_state()) -> tuple:
    with tf.device('/CPU:0'):
        lp = tf.TensorArray(tf.float64, size=n*batch_size, dynamic_size=False)
        E = tf.TensorArray(tf.float64, size=n*batch_size, dynamic_size=False)
        m = tf.TensorArray(tf.float64, size=n*batch_size, dynamic_size=False)

        for i in range(n):
            x = model.sample(params.L, batch_size)
            lp2 = model.log_prob(x)
            E2 = ising.energy(x, params)
            m2 = tfm.abs(tfm.reduce_mean(x, axis=(1, 2, 3)))
            with tf.device('/CPU:0'):
                lp = lp.scatter(tf.range(i*batch_size, (i+1)*batch_size), lp2)
                E = E.scatter(tf.range(i*batch_size, (i+1)*batch_size), E2)
                m = m.scatter(tf.range(i*batch_size, (i+1)*batch_size), m2)

        with tf.device('/CPU:0'):
            lp = lp.stack()
            E = E.stack()
            m = m.stack()
            rand = g.uniform([n*batch_size], dtype=tf.float64)

            acc = 0
            ptr = 0
            mag = tf.TensorArray(tf.float64, size=n*batch_size, dynamic_size=False)
            ener = tf.TensorArray(tf.float64, size=n*batch_size, dynamic_size=False)
            mag = mag.write(0, m[ptr])
            ener = ener.write(0, E[ptr])
            for j in range(1, n*batch_size):
                arg = beta*(E[ptr] - E[j]) + (lp[ptr] - lp[j])
                if rand[j] < tfm.exp(arg):
                    ptr = j
                    acc += 1
                mag = mag.write(j, m[ptr])
                ener = ener.write(j, E[ptr])

            tf.print(f'Acceptance ratio: {acc/n*batch_size - 1}')
            return mag.stack().numpy(), ener.stack().numpy()
