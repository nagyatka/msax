from abc import ABC, abstractmethod
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import cma
import pyswarms

from msax.error import sax_error

def sax_objective_fun(params, x_source, m_size, l_1, use_inf=False):
    a = int(np.round(params[0]))
    w = int(np.round(params[1]))
    return np.mean([sax_error(x=x, a=a, w=w, memory_limit=m_size, l_1=l_1, use_inf=use_inf) for x in x_source])


def optimize(objective_func, x_source, m_size, l_1 = 1, mode='cma' , **kwargs):
    """
    Available modes: cma, bipop-cma, local-pso, global-pso

    :param l_1:
    :param objective_func:
    :param x_source:
    :param m_size:
    :param mode:
    :param kwargs:
    :return:
    """

    if mode == 'cma' or mode == 'bipop-cma':
        x0 = kwargs.pop('x0')
        sigma0 = kwargs.pop('sigma0')
        popsize = kwargs.pop('popsize')
        seed = kwargs.pop('seed', None)
        verbose = kwargs.pop('verbose', True)

        if verbose:
            verbose = 3
        else:
            verbose = -1

        return CMAOptimizationResult(
            mode,
            cma.fmin(objective_func,
            x0=x0,
            sigma0=sigma0,
            args=(x_source, m_size, l_1),
            bipop=True if mode=='bipop-cma' else False,
            options={'popsize': popsize, 'seed': seed, 'verbose': verbose}))

    elif mode == 'local-pso' or mode ==  'global-pso':
        n_particles = kwargs.pop('n_particles')
        bounds = ([3.0, 2.0], [np.inf, np.inf])
        options = {
            'c1': kwargs.pop('c1'),
            'c2': kwargs.pop('c2'),
            'w': kwargs.pop('w')
        }
        iters = kwargs.pop('iters')

        min_a, max_a, min_w, max_w = 3.0, 500.0, 2.0, 500.0
        all_a = np.arange(min_a, max_a, max_a / n_particles)
        all_w = np.arange(min_w, max_w, max_w / n_particles)
        init_pos = np.array([all_a, all_w]).transpose()

        def pso_function_wrapper(particle_coords, **params):
            """
            Wrapper function for the objective function because the pso implementation passes all particles in one
            list instead of passing them one by one.

            :param objective_func:
            :param particle_coords:
            :param kwargs:
            :return:
            """

            obj_func_wrapper = partial(objective_func, **params)
            with Pool() as p:
                return np.array(p.map(obj_func_wrapper, particle_coords, chunksize=3))

            # This implementation is slower (+5-10% time)
            #res = [objective_func(particle_coord, **params) for particle_coord in particle_coords]
            #return np.array(res)

        if mode == 'global-pso':
            optimizer = pyswarms.single.GlobalBestPSO(
                n_particles=n_particles,
                dimensions=2,
                options=options,
                init_pos=init_pos,
                bounds=bounds)
            cost, pos = optimizer.optimize(
                pso_function_wrapper,
                iters=iters,
                fast=True,
                x_source=x_source,
                m_size=m_size,
                use_inf=True,
                l_1=l_1)
            return PSOOptimizationResult(mode, cost, pos, iters, optimizer.cost_history)
        else:
            options['k'] = kwargs.pop('k')
            options['p'] = kwargs.pop('p')
            optimizer = pyswarms.single.LocalBestPSO(
                n_particles=n_particles,
                dimensions=2,
                options=options,
                init_pos=init_pos,
                bounds=bounds)
            cost, pos = optimizer.optimize(
                pso_function_wrapper,
                iters=iters,
                fast=True,
                x_source=x_source,
                m_size=m_size,
                use_inf=True,
                l_1=l_1)
            return PSOOptimizationResult(mode, cost, pos, iters, optimizer.cost_history)
    else:
        raise RuntimeError('Unknown optimization mode')


class OptimizationResult(ABC):

    @property
    @abstractmethod
    def optimizer_name(self):
        pass

    @property
    @abstractmethod
    def w(self):
        pass

    @property
    @abstractmethod
    def a(self):
        pass

    @property
    @abstractmethod
    def cost(self):
        pass

    @property
    @abstractmethod
    def iters(self):
        pass

    @property
    @abstractmethod
    def history(self):
        pass

    def __str__(self):
        return "OptimizationResult ({}): w={}, a={}, (value/cost: {}, #iterations: {})".format(
            self.optimizer_name,
            self.w,
            self.a,
            self.cost,
            self.iters)

    def __repr__(self):
        return self.__str__()


class CMAOptimizationResult(OptimizationResult):
    def __init__(self, name, cma_result):
        self.name = name
        self.cma_result = cma_result
        self.hist = cma_result[-1].load().f[:,-1].copy()

    @property
    def optimizer_name(self):
        return self.name

    @property
    def w(self):
        return int(np.round(self.cma_result[0][1]))

    @property
    def a(self):
        return int(np.round(self.cma_result[0][0]))

    @property
    def cost(self):
        return self.cma_result[1]

    @property
    def iters(self):
        return self.cma_result[4]

    @property
    def history(self):
        return self.hist


class PSOOptimizationResult(OptimizationResult):
    def __init__(self, name, cost, pos, iters, hist):
        self.name = name
        self.pso_cost = cost
        self.pos = pos
        self.iter_no = iters
        self.hist = hist

    @property
    def optimizer_name(self):
        return self.name

    @property
    def w(self):
        return np.round(self.pos[1])

    @property
    def a(self):
        return np.round(self.pos[0])

    @property
    def cost(self):
        return self.pso_cost

    @property
    def iters(self):
        return self.iter_no

    @property
    def history(self):
        return self.hist





