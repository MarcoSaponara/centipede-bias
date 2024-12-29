import numpy as np

from egttools.games import AbstractTwoPLayerGame

class CentipedeGame(AbstractTwoPLayerGame):
    
    def __init__(self, 
                 payoffs_pl1 : np.ndarray,
                 payoffs_pl2 : np.ndarray,
                 strategies : np.ndarray,
                ):
        self.payoffs_pl1 = payoffs_pl1
        self.payoffs_pl2 = payoffs_pl2
        assert len(payoffs_pl1) == len(payoffs_pl2)
        
        self.nb_steps = len(payoffs_pl1) - 1
        self.nb_actions = self.nb_steps//2 + 1
        
        self.strategies = strategies
        self.strategies_ = strategies
        self.nb_strategies_ = len(strategies)
        
        AbstractTwoPLayerGame.__init__(self, len(strategies))
            
    
    def get_min_take(self,
                     take_pdf_pl1 : np.ndarray,
                     take_pdf_pl2 : np.ndarray,
                    ) -> np.ndarray:
        # P(min(X,Y)>a) = P(X>a, Y>a) = P(X>a)P(Y>a)
        # P(min(X,Y)>a) = 1 - P(min(X,Y)<=a)
        # P(min(X,Y)<=a) = 1 - P(X>a)P(Y>a)
        #
        # P(min(X,Y)<=0) = X[0]
        # P(min(X,Y)<=1) = 1 - sum(X[2:]) * sum(Y[2:])
        # ...
        # P(min(X,Y)<=nb_steps) = 1
        
        assert len(take_pdf_pl1)==len(take_pdf_pl2)
        assert len(take_pdf_pl1)==self.nb_steps+1
        
        min_take_cdf = np.zeros(self.nb_steps+1, dtype = float)
        
        min_take_cdf[0] = take_pdf_pl1[0]
        for i in range(1, self.nb_steps):
            min_take_cdf[i] = 1. - take_pdf_pl1[i+1:].sum() * take_pdf_pl2[i+1:].sum()
        min_take_cdf[-1] = 1.
        
        min_take_pdf = np.ediff1d(min_take_cdf, to_begin = min_take_cdf[0])
        assert np.isclose(min_take_pdf.sum(), 1.)
       
        return min_take_pdf
    
    
    def zero_padding_pl1(self, p : np.ndarray) -> np.ndarray:
        # [p_0, 0, p_2, ..., 0, p_n]
        return np.insert(p, slice(1, None), 0.)
    
    def zero_padding_pl2(self, p : np.ndarray) -> np.ndarray:
        # [0, p_1, ..., p_{n-1}, p_n]
        tmp = np.insert(p[:-1], slice(1, None), 0.)
        return np.array([0.]+list(tmp)+[p[-1]], dtype = float)
    
    
    def get_take_distributions(self) -> np.ndarray:
        take_distribution_matrix = np.zeros((self.nb_strategies_, self.nb_strategies_, self.nb_steps+1), dtype = float)
        
        for i, strategy_a in enumerate(self.strategies):
            assert np.isclose(strategy_a[:self.nb_actions].sum(),1.) and np.isclose(strategy_a[self.nb_actions:].sum(),1.)

            p_a_as_pl1 = self.zero_padding_pl1(strategy_a[:self.nb_actions])
            p_a_as_pl2 = self.zero_padding_pl2(strategy_a[self.nb_actions:])
            
            for j, strategy_b in enumerate(self.strategies):
                assert np.isclose(strategy_b[:self.nb_actions].sum(),1.) and np.isclose(strategy_b[self.nb_actions:].sum(),1.)

                # A = pl.1, B = pl.2
                p_b_as_pl2 = self.zero_padding_pl2(strategy_b[self.nb_actions:])
                
                # A = pl.2, B = pl.1
                p_b_as_pl1 = self.zero_padding_pl1(strategy_b[:self.nb_actions])
                
                take_distribution_matrix[i,j]=self.get_min_take(p_a_as_pl1,p_b_as_pl2)+self.get_min_take(p_b_as_pl1,p_a_as_pl2)
     
        return .5 * take_distribution_matrix
    
    
    def get_unconditional_take_distributions(self) -> np.ndarray:
        # not conditional on roles
        take_distribution_matrix = np.zeros((self.nb_strategies_, self.nb_strategies_, self.nb_steps+1), dtype = float)
        
        for i, strategy_a in enumerate(self.strategies):
            assert np.isclose(strategy_a[:self.nb_actions].sum(), 1.) and np.isclose(strategy_a[self.nb_actions:].sum(), 1.)

            p_a_as_pl1 = self.zero_padding_pl1(strategy_a[:self.nb_actions])
            p_a_as_pl2 = self.zero_padding_pl2(strategy_a[self.nb_actions:])
            
            for j, strategy_b in enumerate(self.strategies):
                assert np.isclose(strategy_b[:self.nb_actions].sum(), 1.) and np.isclose(strategy_b[self.nb_actions:].sum(), 1.)

                # A = pl.1, B = pl.2
                p_b_as_pl2 = self.zero_padding_pl2(strategy_b[self.nb_actions:])
                
                # A = pl.2, B = pl.1
                p_b_as_pl1 = self.zero_padding_pl1(strategy_b[:self.nb_actions])
                
                take_distribution_matrix[i,j]= self.get_min_take(.5 * (p_a_as_pl1 + p_a_as_pl2), .5 * (p_b_as_pl1 + p_b_as_pl2))
     
        return take_distribution_matrix
    
    

    def calculate_payoffs_pl1(self) -> np.ndarray:
        
        payoffs_pl1 = np.zeros((self.nb_strategies_, self.nb_strategies_), dtype = float)
        
        for i, strategy_a in enumerate(self.strategies):

            p_a_as_pl1 = strategy_a[:self.nb_actions]
            assert np.isclose(p_a_as_pl1.sum(), 1.)
            p_a_as_pl1 = self.zero_padding_pl1(p_a_as_pl1)
            
            for j, strategy_b in enumerate(self.strategies):
                
                p_b_as_pl2 = strategy_b[self.nb_actions:]
                assert np.isclose(p_b_as_pl2.sum(), 1.)
                p_b_as_pl2 = self.zero_padding_pl2(p_b_as_pl2)
                    
                take = self.get_min_take(p_a_as_pl1, p_b_as_pl2) # A = pl.1, B = pl.2
                payoffs_pl1[i,j] = take @ self.payoffs_pl1
                
        return payoffs_pl1
    
    def calculate_payoffs_pl2(self) -> np.ndarray:
        
        payoffs_pl2 = np.zeros((self.nb_strategies_, self.nb_strategies_), dtype = float)
        
        for i, strategy_a in enumerate(self.strategies):

            p_a_as_pl2 = strategy_a[self.nb_actions:]
            assert np.isclose(p_a_as_pl2.sum(), 1.)
            p_a_as_pl2 = self.zero_padding_pl2(p_a_as_pl2)

            for j, strategy_b in enumerate(self.strategies):
                
                p_b_as_pl1 = strategy_b[:self.nb_actions]
                assert np.isclose(p_b_as_pl1.sum(), 1.)
                p_b_as_pl1 = self.zero_padding_pl1(p_b_as_pl1)

                take = self.get_min_take(p_b_as_pl1, p_a_as_pl2) # A = pl.2, B = pl.1
                payoffs_pl2[i,j] = take @ self.payoffs_pl2

        return payoffs_pl2
    
    
    def calculate_payoffs(self) -> np.ndarray:
        # \pi_{A}(A vs B) = (\pi_A(A as Pl.1, B as Pl.2) + \pi_A(B as Pl.1, A as Pl.2))/2
        pi1 = self.calculate_payoffs_pl1()
        pi2 = self.calculate_payoffs_pl2()
        #print(pi1, pi2)
        self.payoffs_ = .5 * (pi1 + pi2)

        return self.payoffs()
    
    
    def get_normal_form(self) -> np.ndarray:
        A = np.zeros((self.nb_actions, self.nb_actions), dtype = float)
        B = np.zeros((self.nb_actions, self.nb_actions), dtype = float)
        
        for i in range(self.nb_actions):
            for j in range(self.nb_actions):
                take = min(min(2*i, self.nb_steps), min(2*j + 1, self.nb_steps))
                A[i,j] = self.payoffs_pl1[take]
                B[i,j] = self.payoffs_pl2[take]
                
        return A,B
            