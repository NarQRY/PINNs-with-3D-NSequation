import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
# from plotting import newfig, savefig

np.random.seed(1234)


# tf.random.set_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, z, t, u, v, w, phi, layers):

        X = np.concatenate([x, y, z, t], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.x = x
        self.y = y
        self.z = z
        self.t = t

        self.u = u
        self.v = v
        self.w = w

        self.phi = phi  # Adding phi to the class

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)

        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.w_tf = tf.placeholder(tf.float32, shape=[None, self.w.shape[1]])
        self.phi_tf = tf.placeholder(tf.float32, shape=[None, self.phi.shape[1]])

        # self.u_pred, self.v_pred, self.w_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred = self.net_NS(self.x_tf, self.y_tf, self.z_tf, self.t_tf)
        self.u_pred, self.v_pred, self.w_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred, self.phi_pred = self.net_NS(self.x_tf, self.y_tf, self.z_tf, self.t_tf)

        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.w_tf - self.w_pred)) + \
                    tf.reduce_sum(tf.square(self.phi_tf - self.phi_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_w_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))

        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # Normalize Coordinates and Velocities to [-1,1]
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, y, z, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        U_and_phi = self.neural_net(tf.concat([x,y,z,t], 1), self.weights, self.biases)
        u = U_and_phi[:,0:1]
        v = U_and_phi[:,1:2]
        w = U_and_phi[:,2:3]
        phi = U_and_phi[:,3:4]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]

        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]

        # f_u = u_t + lambda_1*(u*u_x + v*u_y + w*u_z) + p_x - lambda_2*(u_xx + u_yy + u_zz)
        # f_v = v_t + lambda_1*(u*v_x + v*v_y + w*v_z) + p_y - lambda_2*(v_xx + v_yy + v_zz)
        # f_w = w_t + lambda_1*(u*w_x + v*w_y + w*w_z) + p_z - lambda_2*(w_xx + w_yy + w_zz)

        f_u = u_t + lambda_1*(u*u_x + v*u_y + w*u_z) - lambda_2*(u_xx + u_yy + u_zz)
        f_v = v_t + lambda_1*(u*v_x + v*v_y + w*v_z) - lambda_2*(v_xx + v_yy + v_zz)
        f_w = w_t + lambda_1*(u*w_x + v*w_y + w*w_z) - lambda_2*(w_xx + w_yy + w_zz)

        return u, v, w, f_u, f_v, f_w, phi
        # return u, v, w, p, f_u, f_v, f_w

    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v, self.w_tf: self.w, self.phi_tf: self.phi}

        start_time = time.time()
        LossOutput = open('ItLossLamda.txt', 'w')
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                LossOutput.writelines('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f\n' %
                      (it, loss_value, lambda_1_value, lambda_2_value))
                start_time = time.time()

        # self.optimizer.minimize(self.sess,
        #                         feed_dict = tf_dict,
        #                         fetches = [self.loss, self.lambda_1, self.lambda_2],
        #                         loss_callback = self.callback)


    def predict(self, x_star, y_star, z_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        # p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, w_star, phi_star
        # return u_star, v_star, w_star, p_star


def plot_solution(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    z = np.linspace(lb[2], ub[2], nn)
    X, Y, Z = np.meshgrid(x, y, z)

    U_star = griddata(X_star, u_star.flatten(), (X, Y, Z), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, Z, U_star, cmap='jet')
    plt.colorbar()


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":

    N_train = 5000

    Re = 0.02

    dom_x = 12
    dom_y = 12
    dom_z = 40

    layers = [4, 20, 20, 20, 20, 20, 4]  # 4 Inputs: x, y, z, t; 4 Outputs: u, v, w, phi

    # Load Data
    data = scipy.io.loadmat('../Stokes.mat')

    U_star = data['U_star']  # N x 3 x T
    # P_star = data['p_star'] # N x T
    Phi_star = data['Phi_star']  # N x 1 x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 3

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    ZZ = np.tile(X_star[:, 2:3], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    WW = U_star[:, 2, :]  # N x T
    PPHI = Phi_star  # N x T
    # PP = P_star # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    z = ZZ.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    w = WW.flatten()[:, None]  # NT x 1
    phi = PPHI.flatten()[:, None]  # NT x 1
    # p = PP.flatten()[:,None] # NT x 1

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    z_train = z[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]
    w_train = w[idx, :]
    phi_train = phi[idx, :]

    # Training
    model = PhysicsInformedNN(x_train, y_train, z_train, t_train, u_train, v_train, w_train, phi_train, layers)
    model.train(50000)

    # Test Data
    snap = np.array([1])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    z_star = X_star[:, 2:3]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    w_star = U_star[:, 2, snap]
    phi_star = Phi_star[:, snap]
    # p_star = P_star[:,snap]

    # Prediction
    u_pred, v_pred, w_pred, phi_pred = model.predict(x_star, y_star, z_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    error_phi = np.linalg.norm(phi_star - phi_pred, 2) / np.linalg.norm(phi_star, 2)
    # error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - 1.0 / Re) / (1.0 / Re) * 100

    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error w: %e' % (error_w))
    print('Error phi: %e' % (error_phi))
    # print('Error p: %e' % (error_p))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))

    # Save Results
    VelOutput = open('VelOutput.vtk', 'w')
    VelOutput.write('# vtk DataFile Version 2.0\n')
    VelOutput.write('Generated by Q RY\n')
    VelOutput.write('ASCII\n')
    VelOutput.write('DATASET STRUCTURED_POINTS\n')
    VelOutput.write('DIMENSIONS %d %d %d\n' % (dom_x, dom_y, dom_z))
    VelOutput.write('ORIGIN 0 0 0\n')
    VelOutput.write('SPACING 1 1 1\n')
    VelOutput.write('POINT_DATA %d\n' % (dom_x * dom_y * dom_z))
    VelOutput.write('VECTORS velocity_field float \n')
    for i in range(dom_x * dom_y * dom_z):
        VelOutput.writelines('%e %e %e\n' % (u_pred[i, 0], v_pred[i, 0], w_pred[i, 0]))
    VelOutput.close()

    PhiOutput = open('PhiOutput.vtk', 'w')
    PhiOutput.write('# vtk DataFile Version 2.0\n')
    PhiOutput.write('Generated by Q RY\n')
    PhiOutput.write('ASCII\n')
    PhiOutput.write('DATASET STRUCTURED_POINTS\n')
    PhiOutput.write('DIMENSIONS %d %d %d\n' % (dom_x, dom_y, dom_z))
    PhiOutput.write('ORIGIN 0 0 0\n')
    PhiOutput.write('SPACING 1 1 1\n')
    PhiOutput.write('POINT_DATA %d\n' % (dom_x * dom_y * dom_z))
    PhiOutput.write('SCALARS composition float 1\n')
    PhiOutput.write('LOOKUP_TABLE default\n')
    for i in range(dom_x * dom_y * dom_z):
        PhiOutput.writelines('%e\n' % (phi_pred[i, 0]))
    PhiOutput.close()
