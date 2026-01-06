#!/usr/bin/env python
# --------------------------------------------------------------%
# Created by "Mustafa Surhay Samsa"            %
# Email: samsasurhay@gmail.com                                  %
# --------------------------------------------------------------%
#
# Links:
#     1. https://www.sciencedirect.com/science/article/pii/S0950705124003721
#
# References:
#     [1] Wang, X., Snášel, V., Mirjalili, S., Pan, J.-S., Kong, L., & Shehadeh, H. A. (2024).
#         Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm
#         for engineering optimization. Knowledge-Based Systems, 295, 111737.
#         https://doi.org/10.1016/j.knosys.2024.111737

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAPO(Optimizer):
    """
    Artificial Protozoa Optimizer (APO) - Original Version

    * Üç temel davranış formu içerir:
        - Dormancy (dinlenme formu)
        - Reproduction (üreme formu)
        - Foraging (beslenme formu: autotroph + heterotroph)

    Bu implementasyon:
        - Orijinal APO denklemlerini yakından takip eder.
        - MEALPY'nin Optimizer arayüzü ile tam uyumludur.
        - Sürekli (continuous) optimizasyon problemleri için tasarlanmıştır.

    Hyper-parameters:
        - pf_max (float): maksimum proportion fraction (0 < pf_max <= 1)
        - np_val (int): foraging aşamasında kullanılan neighbor pairs sayısı

    Örnek kullanım (generic continuous problem):

    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from mealpy.bio_based import APO
    >>>
    >>> def sphere(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="x"),
    >>>     "obj_func": sphere,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = APO.OriginalAPO(epoch=1000, pop_size=50, pf_max=0.1, np_val=1)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pf_max: float = 0.1,
        np_val: int = 1,
        **kwargs: object,
    ) -> None:
        """
        Args:
            epoch   (int): maksimum iterasyon sayısı
            pop_size(int): popülasyon boyutu
            pf_max (float): maksimum proportion fraction
            np_val  (int): neighbor pair sayısı
        """
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pf_max = self.validator.check_float("pf_max", pf_max, (0.0, 1.0))
        self.np_val = self.validator.check_int("np_val", np_val, [1, 10])

        self.set_parameters(["epoch", "pop_size", "pf_max", "np_val"])

        # APO kendi içinde popülasyonu sıraladığı için Optimizer'ın otomatik sort’una ihtiyaç yok.
        self.sort_flag = False

 
    # Yardımcı fonksiyonlar
    
    def _rand_position(self) -> np.ndarray:
        """Arama uzayında [lb, ub] aralığında rastgele bir konum üretir."""
        u = self.generator.random(self.problem.n_dims)
        return self.problem.lb + u * (self.problem.ub - self.problem.lb)

 
    # Ana APO iterasyonu

    def evolve(self, epoch: int = None):
        """
        APO algoritmasının ana denklemleri.

        Args:
            epoch (int): 0-based iterasyon index'i (0..self.epoch-1)
        """
        if epoch is None:
            epoch = 0

        ps = self.pop_size
        dim = self.problem.n_dims

        # Mevcut popülasyonu fitness'a göre sırala
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)

        # Agent -> numpy array
        protozoa = np.array([agent.solution for agent in pop_sorted])
        protozoa_fit = np.array([agent.target.fitness for agent in pop_sorted])

        new_protozoa = protozoa.copy()
        epn = np.zeros((self.np_val, dim))

        # Orijinal makaledeki gibi 1..T aralığını epoch'a mapliyoruz
        t = epoch + 1
        T_max = max(1, self.epoch)

        # Proportion fraction
        pf = self.pf_max * self.generator.random()
        num_dormancy = int(np.ceil(ps * pf))
        if num_dormancy > 0:
            ri = set(self.generator.permutation(ps)[:num_dormancy])
        else:
            ri = set()

        # Her protozoa için form seçimi ve güncelleme
        for i in range(ps):
            # Dormancy / Reproduction form
            if i in ri:
                # pdr: dormancy olasılığı
                pdr = 0.5 * (1.0 + np.cos((1.0 - i / ps) * np.pi))

                if self.generator.random() < pdr:
                    # Dormancy form: tamamen rastgele yeni konum (keşif)
                    pos_new = self._rand_position()
                else:
                    # Reproduction form
                    flag = self.generator.choice([-1.0, 1.0])
                    mr = np.zeros(dim)

                    # Hangi boyutların aktif olacağı
                    num_dims = int(np.ceil(self.generator.random() * dim))
                    if num_dims > 0:
                        sel = self.generator.permutation(dim)[:num_dims]
                        mr[sel] = 1.0

                    rand_pos = self._rand_position()
                    pos_new = protozoa[i] + flag * self.generator.random() * rand_pos * mr

                new_protozoa[i] = pos_new

            # Foraging form (autotroph / heterotroph)
            else:
                # f: beslenme faktörü (iterasyona bağlı)
                f = self.generator.random() * (1.0 + np.cos(t / T_max * np.pi))

                mf = np.zeros(dim)
                num_dims = int(np.ceil(dim * i / ps))
                if num_dims > 0:
                    sel = self.generator.permutation(dim)[:num_dims]
                    mf[sel] = 1.0

                # pah: autotroph vs heterotroph seçimi
                pah = 0.5 * (1.0 + np.cos(t / T_max * np.pi))

                # ------------------ Autotroph form -------------------
                if self.generator.random() < pah:
                    # j: rastgele seçilen bir protozoa
                    j = self.generator.integers(0, ps)

                    for k in range(self.np_val):
                        if i == 0:
                            km = 0
                            kp = self.generator.integers(1, ps)
                        elif i == ps - 1:
                            km = self.generator.integers(0, ps - 1)
                            kp = ps - 1
                        else:
                            km = self.generator.integers(0, i)
                            kp = i + self.generator.integers(1, ps - i)

                        denom = protozoa_fit[kp] + self.EPSILON
                        wa = np.exp(-np.abs(protozoa_fit[km] / denom))
                        epn[k] = wa * (protozoa[km] - protozoa[kp])

                    mean_epn = np.mean(epn, axis=0)
                    pos_new = protozoa[i] + f * (protozoa[j] - protozoa[i] + mean_epn) * mf

                # ----------------- Heterotroph form ------------------
                else:
                    for k in range(self.np_val):
                        if i == 0:
                            imk = 0
                            ipk = i + k
                        elif i == ps - 1:
                            imk = ps - 1 - k
                            ipk = ps - 1
                        else:
                            imk = i - k
                            ipk = i + k

                        imk = max(0, min(imk, ps - 1))
                        ipk = max(0, min(ipk, ps - 1))

                        denom = protozoa_fit[ipk] + self.EPSILON
                        wh = np.exp(-np.abs(protozoa_fit[imk] / denom))
                        epn[k] = wh * (protozoa[imk] - protozoa[ipk])

                    # x_near: iterasyona göre daralan yakın komşu bölgesi
                    flag_vec = self.generator.choice([-1.0, 1.0], size=dim)
                    shrink = 1.0 - t / T_max
                    rand_vec = self.generator.random(dim)
                    x_near = (1.0 + flag_vec * rand_vec * shrink) * protozoa[i]

                    mean_epn = np.mean(epn, axis=0)
                    pos_new = protozoa[i] + f * (x_near - protozoa[i] + mean_epn) * mf

                new_protozoa[i] = pos_new

        # MEALPY Agent güncellemesi (greedy selection)
        pop_new = []
        for idx in range(ps):
            pos_new = self.correct_solution(new_protozoa[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                # Tek çekirdek: fitness hemen hesaplanır ve greedy selection yapılır
                agent.target = self.get_target(pos_new)
                pop_sorted[idx] = self.get_better_agent(
                    agent, pop_sorted[idx], self.problem.minmax
                )

        if self.mode in self.AVAILABLE_MODES:
            # Paralel modlarda: pop_new için toplu fitness hesabı ve greedy selection
            pop_new = self.update_target_for_population(pop_new)
            pop_sorted = self.greedy_selection_population(
                pop_sorted, pop_new, self.problem.minmax
            )

        # Son popülasyon
        self.pop = self.get_sorted_population(pop_sorted, self.problem.minmax)
