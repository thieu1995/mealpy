import numpy as np
from mealpy.optimizer import Optimizer

class OriginalGAO(Optimizer):
    """
    Green Anaconda Optimization (GAO)
    
    Links:
        1. https://www.mdpi.com/2313-7673/8/3/121 (Original Paper)
        
    Notes:
        1. The implementation follows the equations provided in the paper.
        2. Phase 1: Mating Season (Exploration)
        3. Phase 2: Hunting Strategy (Exploitation)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of Green Anaconda Optimization (GAO).
        
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        
        # Fitness değerlerini ve pozisyonları listeden çekme (Vektörizasyon için hazırlık)
        # Mealpy'de self.pop bir Agent nesneleri listesidir.
        fitness_list = np.array([agent.target.fitness for agent in self.pop])
        pos_list = np.array([agent.solution for agent in self.pop])
        
        for i in range(self.pop_size):
            agent = self.pop[i]
            
            # --- FAZ 1: Mating Season (Exploration) ---
            # Kendisinden daha iyi olanları bul (Minimizasyon problemi)
            better_indices = np.where(fitness_list < fitness_list[i])[0]
            
            # Geçici pozisyon (Phase 1 sonucu)
            pos_new_p1 = agent.solution.copy()
            
            if len(better_indices) > 0:
                CFF = fitness_list[better_indices]
                CFF_max = np.max(CFF)
                
                # Olasılık Hesabı (Eq. 5)
                diffs = CFF - CFF_max
                denominator = np.sum(diffs)
                
                if denominator == 0:
                    probs = np.ones(len(CFF)) / len(CFF)
                else:
                    probs = diffs / denominator
                
                cum_probs = np.cumsum(probs)
                r_rand = np.random.rand()
                
                selected_local_idx = 0
                for idx, cp in enumerate(cum_probs):
                    if r_rand <= cp:
                        selected_local_idx = idx
                        break
                
                selected_global_idx = better_indices[selected_local_idx]
                SF = pos_list[selected_global_idx] # Selected Female Position
                
                # Eq. 8
                I = np.random.randint(1, 3, size=self.problem.n_dims)
                r = np.random.rand(self.problem.n_dims)
                
                # Yeni pozisyon hesabı
                pos_new_p1 = agent.solution + r * (SF - I * agent.solution)
                
                # Sınır kontrolü (Mealpy fonksiyonu ile)
                pos_new_p1 = self.problem.correct_solution(pos_new_p1)
                
                # Greedy Selection (Phase 1)
                # Mealpy'de evaluate anlık yapılır veya pop_new'e atılıp toplu yapılır.
                # Burada lojik gereği adım adım gidiyoruz.
                agent_new_p1 = self.generate_agent(pos_new_p1)
                if self.compare_agent(agent_new_p1, agent):
                    pos_new_p1 = agent_new_p1.solution
            
            # --- FAZ 2: Hunting Strategy (Exploitation) ---
            # Eq. 10
            # t mevcut epoch'tur. Mealpy'de epoch 1'den başlar (args'dan gelir)
            r2 = np.random.rand(self.problem.n_dims)
            pos_new_p2 = pos_new_p1 + (1 - 2 * r2) * (self.problem.ub - self.problem.lb) / epoch
            
            # Sınır Kontrolü
            pos_new_p2 = self.problem.correct_solution(pos_new_p2)
            
            # Yeni ajanı oluştur ve listeye ekle
            # Mealpy'de greedy selection genellikle döngü sonunda yapılır ama 
            # GAO her adımda greedy yaptığı için en son halini ekliyoruz.
            agent_new_p2 = self.generate_agent(pos_new_p2)
            pop_new.append(agent_new_p2)

        # Mevcut popülasyonu güncelle (Greedy Selection mantığı Mealpy update metodunda vardır)
        self.pop = self.update_target_wrapper_population(self.pop, pop_new)