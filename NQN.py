import numpy
import copy
import sys


def fmin_l_bfgs_b(funObj, x, gradObj, bounds=None, m=20, M=1, pgtol=1e-5,
                  iprint=-1, maxfun=15000, maxiter=15000, callback=None, factr=0.):
    '''
    Termination Flag:
    0 => Converged
    1 => Reached Maximum Iterations
    2 => Reached Maximum Function Evaluations
    3 => Converged to Nonstationary Point
    4 => Abnormal Termination in Line Search
    5 => Iterate Has NaN values
    6 => Numerical Issues
    7 => FACTR Convergence
    '''

    n = x.shape[0]
    f = funObj(x)
    g = gradObj(x)
    nCorrections = 0
    multimodel = False
    qn = quasi_newton(n, m, multimodel, g)
    if bounds is None:
        bounds = [(-numpy.inf, numpy.inf) for i in range(n)]
    bnds = bound_helper(bounds)
    x_old = numpy.copy(x)
    g_old = numpy.copy(g)
    f_old = numpy.copy(f)
    func_calls = 1
    termination_flag = 1
    alpha = '-'

    # Options used for variants of NQN
    correct = True
    gradalso = True
    return_hist = False

    GPacket = []
    XPacket = []
    fga_hist = []

    # Calculates \epsilon-MNSG
    def gtilde(x, GPacket, XPacket):
        if M == 1 or len(GPacket) < M:
            return GPacket[-1]
        _GPacket = []

        for i in range(len(GPacket)):
            _GPacket.append(GPacket[i])
        if not _GPacket:
            return GPacket[-1]
        grad_samp_grad = numpy.array(
            sampled_quality_cvx(x, gradObj, packet=_GPacket)[1])
        return grad_samp_grad

    for it in range(maxiter):
        if callback is not None:
            callback(x)
        GPacket.append(g)
        XPacket.append(x)

        while(len(GPacket) > M):
            GPacket.pop(0)
            XPacket.pop(0)

        grad_samp_grad = gtilde(x, GPacket, XPacket)
        pg = bnds.T_omega(grad_samp_grad, x)

        if(iprint > -1):
            print 'Iteration', it, 'NFE', func_calls, 'Function Value:', f, '||g||', numpy.linalg.norm(pg, numpy.inf), 'Active', len(bnds.activity(x, grad_samp_grad)), 'AtBounds', len(bnds.atbounds(x)), 'alpha', alpha

        if return_hist:
            fga_hist.append((f, numpy.linalg.norm(
                g, numpy.inf), len(bnds.atbounds(x))))
        if(numpy.linalg.norm(pg) / numpy.sqrt(n) < pgtol):
            termination_flag = 0
            if iprint > -1:
                print 'Converged!'
            break

        if(func_calls > maxfun):
            if iprint > -1:
                print 'Maximum number of function values reached'
            termination_flag = 2
            break

        gs_activity = bnds.activity(x, grad_samp_grad)
        if gradalso:
            gn_activity = bnds.activity(x, g)
        else:
            gn_activity = set([])
        prev_active = len(set(gn_activity).union(set(gs_activity)))
        flag = True
        orip = qn.multiply(g, set(gn_activity).union(set(gs_activity)))[0]
        while flag:
            p = qn.multiply(g, set(gn_activity).union(set(gs_activity)))
            func_calls += len(p) * multimodel
            if multimodel:
                mm_functions = [f_and_g(x - p[i])[0] for i in range(len(p))]
                p = -p[numpy.argmin(mm_functions)]
            else:
                assert(len(p) == 1)
                p = -p[0]
            wrong = numpy.where(bnds.T_omega(-p, x) != -p)[0]
            if len(wrong) == 0 or (not correct):
                flag = False
            else:
                nCorrections += len(wrong)
                gn_activity = set(gn_activity).union(set(wrong))

        max_step = bnds.alpha_per_coordinate(x, p)
        if not max_step:
            termination_flag = 3
            if iprint > -1:
                print 'Convergence to non-stationary point. Exiting!'
            break
        if numpy.dot(bnds.T_omega(-p, x), g) < 0:
            if iprint > -1:
                print 'Not a feasible descent direction. Exiting!'
            termination_flag = 3
            break
        bnds.betamax = max(max_step)

        alpha, p_to_take, f, g, ls_iter, ls_flag = WeakWolfe(
            x, p, funObj, gradObj, f, g, bnds)
        func_calls += ls_iter
        if alpha == 0.:
            if iprint > -1:
                print 'Acceptable step-size not found'
            termination_flag = 4
            break

        if ls_flag == 1 and iprint > -1:
            print 'Armijo-Wolfe Bracketing Failed. Accepting Armijo-only point.'

        if it > 0 and (f_old - f) / (max(abs(f), abs(f_old), 1.)) <= factr * 1E-16:
            termination = 7
            if iprint > -1:
                print 'FACTR convergence'
            break

        x = x_old + p_to_take

        if numpy.any(numpy.isnan(x)):
            x = x_old
            g = g_old
            f = f_old
            termination_flag = 5
            if iprint > -1:
                print 'Iterate was NaN; rolling back'
            break
        else:
            x = bnds.projection(x)

        y = numpy.copy(g - g_old)
        s = numpy.copy(x - x_old)
        nrm_y = numpy.linalg.norm(y)
        nrm_s = numpy.linalg.norm(s)

        if (nrm_y > 0) and (nrm_s > 0) and (
                numpy.dot(s, y) / nrm_s / nrm_y > 1E-8):
            qn.update(y, s)
        else:
            pass

        g_old = numpy.copy(g)
        x_old = numpy.copy(x)
        f_old = numpy.copy(f)

    return_dict = {}
    return_dict['warnflag'] = termination_flag
    return_dict['grad'] = g
    return_dict['funcalls'] = func_calls
    return_dict['nit'] = it
    if return_hist:
        return_dict['history'] = numpy.array(fga_hist)
    return x, f, return_dict


def WeakWolfe(x, p, funObj, gradObj, f, g, bnds, t=1.):
    '''
    LS FLAG:
    0 => Arimjo-Wolfe Point Found
    1 => Bisection Failed
    2 => Step Too Small
    '''
    betamax = bnds.betamax
    projection = bnds.projection

    alpha = 0.0
    beta = betamax
    t = min(max(t, 1E-8), betamax)
    eps1 = 0.
    eps2 = 1e-6
    c1 = 1e-8
    c2 = 0.9
    f_old = f
    g_old = g
    x0 = numpy.copy(x)

    stored_func = f_old
    stored_grad = g_old
    stored_step = alpha

    ls_iter = 0
    ls_flag = 0
    msg = 'Begin'

    while(1):
        alpha_old = alpha
        beta_old = beta

        x = projection(x0 + (t * p))
        current_func = funObj(x)
        current_grad = gradObj(x)

        ls_iter += 1

        if current_func - f_old >= -c1 * t * \
                numpy.dot(bnds.T_omega(-p, x0), g_old):
            beta = t
            msg = 'Armijo Failure'
        elif -numpy.dot(bnds.T_omega(-p, x), current_grad) <= -c2 * numpy.dot(bnds.T_omega(-p, x0), g_old):
            alpha = t
            stored_step = t
            stored_func = current_func
            stored_grad = current_grad
            msg = 'Wolfe Failure'
        else:
            msg = 'Success'
            break

        if (beta < betamax) or (beta == betamax and alpha == 0):
            t = (alpha + beta) / 2.0
        else:
            t = min(2 * alpha, beta)
        if numpy.isclose(beta, alpha, eps2, eps1):
            ls_flag = 1
            t = stored_step
            current_func = stored_func
            current_grad = stored_grad
            break

    return t, t * p, current_func, current_grad, ls_iter, ls_flag


def sampled_quality_cvx(x, gradObj, perturb=1e-6, packet=None, np=2):
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    if packet is None:
        packet = []
        for sm in range(x.shape[0] * np):
            try_x = x + numpy.random.uniform(-perturb, perturb, x.shape[0])
            packet.append(gradObj(try_x))
        packet.append(gradObj(x))
    packet = numpy.array(packet).T
    coeff_for_stability = numpy.mean(abs(packet))
    G = packet / coeff_for_stability
    _P = matrix(numpy.dot(G.T, G), tc='d')
    _A = matrix(numpy.ones((1, G.shape[1])), tc='d')
    _b = matrix(numpy.ones(1), tc='d')
    _G = matrix(numpy.eye(G.shape[1]) * -1, tc='d')
    _h = matrix(numpy.zeros(G.shape[1]), tc='d')
    _q = matrix(numpy.zeros(G.shape[1]), tc='d')
    try:
        sol = solvers.qp(_P, _q, _G, _h, _A, _b)
        return sol['primal objective'], numpy.dot(
            G, sol['x']).reshape(G.shape[0]) * coeff_for_stability
    except:
        return -1., packet[:, -1]


class quasi_newton:

    def __init__(self, n, m, mm, g):
        self.mem = m
        self.n = n
        self.mem_used = 0
        self.memory_for_s = []
        self.memory_for_y = []
        self.multimodel = mm
        self.B = numpy.eye(n) * max(1., numpy.linalg.norm(g, numpy.inf))

    def update(self, y, s):
        n = self.n
        if self.mem < numpy.inf:
            self.memory_for_y.append(y)
            self.memory_for_s.append(s)
            if len(self.memory_for_s) > self.mem:
                self.memory_for_s.pop(0)
                self.memory_for_y.pop(0)
            else:
                self.mem_used += 1
        else:
            Bs = numpy.dot(self.B, s)
            self.B = self.B + \
                numpy.outer(y, y) / numpy.dot(y, s) - \
                numpy.outer(Bs, Bs) / numpy.dot(s, Bs)
            self.B = 0.5 * (self.B + self.B.T)

    def multiply(self, g, active_set=None):
        free_set = list(set(range(g.shape[0])).difference(active_set))
        if not free_set:
            return [numpy.zeros(g.shape)]

        if self.mem < numpy.inf:
            memory_for_y = copy.deepcopy(self.memory_for_y)
            memory_for_s = copy.deepcopy(self.memory_for_s)
            direction = []
            theta = max(1.0, numpy.linalg.norm(g, 2))
            for i in range(self.mem_used + 1):
                if(len(memory_for_y) == 0):
                    temp_direction = numpy.zeros(g.shape)
                    temp_direction[free_set] = g[free_set] / theta
                    direction.append(temp_direction)
                    break
                Yk = numpy.array(memory_for_y).T
                Sk = numpy.array(memory_for_s).T

                Wk = numpy.hstack((Yk, theta * Sk))
                Dk = numpy.diag([numpy.dot(memory_for_s[i], memory_for_y[i])
                                 for i in range(len(memory_for_s))])
                Lk = numpy.tril(Sk.T.dot(Yk)) - Dk
                Mk_inv = numpy.vstack(
                    (numpy.hstack((-Dk, Lk.T)), numpy.hstack((Lk, theta * numpy.dot(Sk.T, Sk)))))
                try:
                    Mk = numpy.linalg.inv(Mk_inv)
                except:
                    return [g * 0]

                def Z(x):
                    _r = numpy.zeros(g.shape)
                    _r[free_set] = x
                    return _r

                def ZT(x):
                    return x[free_set]
                rc = ZT(g)
                v = Wk.T.dot(Z(rc))
                v = Mk.dot(v)
                ZTW = numpy.array([ZT(Wk[:, i]) for i in range(Wk.shape[1])]).T
                N = 1 / theta * numpy.dot(ZTW.T, ZTW)
                MN = Mk.dot(N)
                N = numpy.eye(MN.shape[0]) - MN
                try:
                    v = numpy.linalg.solve(N, v)
                except:
                    return [g * 0]

                direction.append(
                    Z(1 / theta * rc + 1 / theta**2 * ZT(Wk.dot(v))))
                if not self.multimodel:
                    break
                memory_for_y.pop(0)
                memory_for_s.pop(0)
            return direction
        else:
            direction = numpy.zeros(g.shape)
            try:
                direction[free_set] = numpy.linalg.solve(
                    self.B[numpy.ix_(free_set, free_set)], g[free_set])
            except:
                return [numpy.zeros(g.shape)]
            return [direction]

    def return_B(self, g):
        if self.mem < numpy.inf:
            if len(self.memory_for_y) > 0:
                my_mult = max(1.0, numpy.linalg.norm(g))
            else:
                my_mult = 1.0 * max(1., numpy.linalg.norm(g))
            B = numpy.eye(self.n) * max(1., numpy.linalg.norm(g))
            for i in range(self.mem_used):
                B = B + numpy.outer(self.memory_for_y[i], self.memory_for_y[i]) / numpy.dot(self.memory_for_y[i], self.memory_for_s[i]) - numpy.dot(
                    B, numpy.dot(numpy.outer(self.memory_for_s[i], self.memory_for_s[i]), B)) / numpy.dot(self.memory_for_s[i], numpy.dot(B, self.memory_for_s[i]))
            return B
        else:
            return self.B


class bound_helper:

    def __init__(self, bounds):
        self.n = len(bounds)
        self.activity_hist = set([])
        self.packet = []
        self.l = numpy.zeros(self.n)
        self.u = numpy.zeros(self.n)
        for i in range(self.n):
            self.l[i], self.u[i] = bounds[i]

    def projection(self, iterate):
        assert(numpy.size(self.l) == numpy.size(iterate))
        proj = numpy.median([self.l, iterate, self.u], 0)
        return proj

    def assert_feasible(self, iterate):
        assert(numpy.alltrue((iterate - self.l >= 0)))
        assert(numpy.alltrue((self.u - iterate >= 0)))

    def T_omega(self, direction, iterate):
        n = self.n
        projected_direction = numpy.zeros(n)
        for i in range(n):
            if(iterate[i] == self.l[i]):
                projected_direction[i] = min(direction[i], 0)
            elif(iterate[i] == self.u[i]):
                projected_direction[i] = max(direction[i], 0)
            else:
                projected_direction[i] = direction[i]
        return projected_direction

    def alpha_per_coordinate(self, x, p):
        n = self.n
        alpha_comp = []
        for i in range(n):
            if(p[i] < 0 and not numpy.equal(x[i], self.l[i])):
                alpha_comp.append((self.l[i] - x[i]) / p[i])
            elif(p[i] > 0 and not numpy.equal(x[i], self.u[i])):
                alpha_comp.append((self.u[i] - x[i]) / p[i])
        return alpha_comp

    def activity(self, iterate, g):
        equality_check = lambda x, y: x == y
        n = self.n
        activity = []
        for i in range(n):
            if(equality_check(iterate[i], self.l[i]) and g[i] >= 0):
                activity.append(i)
            elif(equality_check(iterate[i], self.u[i]) and g[i] <= 0):
                activity.append(i)
        self.activity_hist = self.activity_hist.union(activity)
        return activity

    def atbounds(self, iterate):
        equality_check = lambda x, y: x == y
        return numpy.where(numpy.logical_or(equality_check(
            iterate, self.u), equality_check(iterate, self.l)))[0].tolist()

    def reset_activity(self):
        self.activity_hist = set([])

if __name__ == '__main__':
    print 'Testing N=10 Rosenbrock Function'
    import scipy.optimize
    n = 10
    func = scipy.optimize.rosen
    grad = scipy.optimize.rosen_der
    x0 = numpy.random.randn(n)
    bounds = [(-0.5, 0.5) for i in range(n)]
    print 'SciPy:', scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=grad, bounds=bounds)[:2]
    print 'NQN (base):', fmin_l_bfgs_b(func, x0, grad, bounds)[:2]
    print 'NQN (BFGS, m=Inf):', fmin_l_bfgs_b(func, x0, grad, bounds, m=numpy.inf)[:2]
    print 'NQN (M=20):', fmin_l_bfgs_b(func, x0, grad, bounds, M=20)[:2]
    print 'NQN (callback):', fmin_l_bfgs_b(func, x0, grad, bounds, callback=lambda x: numpy.save('TEST', x))[:2]
