import numpy as np
from scipy.special import gammaln
from math import gamma
from numpy import linalg as LA
from ZGH_BRL_VBVC.helpers import normalize, get_margins
from ZGH_BRL_VBVC.Observation import get_scene_features


class BRLearner:
    """Common base class for Bayesian Reinforcement Learners Parameters
    OBS! Most arguments should be possible to remove
    """

    def __init__(self, n=None,  idx=-1, num_grids=4, fsz=5,
                 isEpsilonGreedy=None, isLearning=None,    type=None, normalize_input=False, depth_dist=None,
                 learningRate_Val=1,   learningRate_Rate=0, learningRate_Final=0,
                 gd=0,        initVar=0.1,  forgettingFactor=0,
                 CN=0,        PN=0,         PAN=0, aveR=0,
                 posSample=0, negSample=0,
                 tmpVal =0,   tmpRate=0,    tmpFinal=0,
                 disVal=0,    disRate=0,    disFinal=0,
                 pxmVal=0,    pxmRate=0,    pxmFinal=0,
                 pmxVal=0,    pmxRate=0,    pmxFinal=0,
                 pxcVal=0,    pxcRate=0,    pxcFinal=0,
                 max_mix=1000):

        self.idx = idx
        self.depth_dist = depth_dist
        if n is None:
            self.n = fsz * num_grids
            if self.depth_dist:
                self.n += 3
        else:
            self.n = n
        self.fsz = fsz
        self.normalize_input    = normalize_input
        self.forgettingFactor   = forgettingFactor
        self.learningRate_Val   = learningRate_Val
        self.learningRate_Rate  = learningRate_Rate
        self.learningRate_Final = learningRate_Final
        self.isLearning       = isLearning
        self.gd               = gd
        self.initVar          = initVar
        self.CN               = CN
        self.PN               = PN
        self.PAN              = PAN
        self.aveR             = aveR

        self.a                = np.zeros((1, max_mix)) # Number of times a specific component(Mj) has been updated
        self.a[:, 0]          = self.n
        self.v                = np.zeros((1, max_mix))
        self.v[:, 0]          = self.n + 1
        self.b                = np.zeros((self.n, self.n, max_mix))
        self.b[:, :, 0]       = self.initVar * np.eye(self.n)
        self.m                = np.zeros((self.n, max_mix))
        self.m[:, 0]          = np.ones(self.n) / self.n
        self.x                = - np.ones((self.n, 1))
        self.nx               = self.x
        self.p                = np.zeros((self.CN, max_mix))
        self.p[:, 0]          = np.ones(self.CN) * 0.1  # np.random.rand(self.CN)

        self.posSample        = posSample
        self.negSample        = negSample
        self.disVal           = disVal
        self.disRate          = disRate
        self.disFinal         = disFinal
        self.dis              = 0.
        self.pxmVal           = pxmVal
        self.pxmRate          = pxmRate
        self.pxmFinal         = pxmFinal
        self.pmxVal           = pmxVal
        self.pmxRate          = pmxRate
        self.pmxFinal         = pmxFinal
        self.pxcVal           = pxcVal
        self.pxcRate          = pxcRate
        self.pxcFinal         = pxcFinal
        self.pcxVal           = 1 / self.CN
        self.pcxRate          = 2e-2
        self.pcxFinal         = 2 / self.CN

        self.log_learningRate = list()
        self.log_TDerr        = list()
        self.log_Reward       = list()
        self.log_aveR         = list()
        self.log_PN           = list()
        self.log_PAN          = list()
        self.log_dist         = list()
        self.log_tmp          = list()
        self.log_gd           = list()
        self.log_cg           = list()
        self.log_dis          = list()
        self.log_pxm          = list()
        self.log_pmx          = list()
        self.log_pcx          = list()
        self.decisions        = [0]

        self.pmc = np.zeros((self.PN + 1, self.CN))
        self.pxc = np.zeros((1, self.CN))
        self.pxm = np.zeros((1, self.PN + 1))
        self.pc = self.p
        self.pm = self.p
        self.pmx = np.zeros((self.PN + 1, 1))
        self.pmcx = np.zeros((self.PN + 1, self.CN))
        self.pcx = np.zeros((self.CN, 1))
        self.pmncx = np.zeros((self.PN + 1, self.CN))

        self.TDerr = 0.
        self.w = np.zeros((self.PN + 1, 1))
        self.Cg = 0
        self.Reward = 0.

        self.type = type
        self.oldTDerr = -5.

        self.tmpVal = tmpVal
        self.tmpRate = tmpRate
        self.tmpFinal = tmpFinal

        self.probw = np.ones((self.CN, 1))
        self.Fusion_ActRew_NexSt = []  # Not used but nice!

        self.calculate_pm_pc_pmc()
        self.percepReward = 0
        self.num_grids = num_grids
        self.ind_dis = 0

    def update_state(self, image, depth_image, segment_image):
        """
        Update internal state
        inputs: image for Agents
        """
        margins = get_margins()
        self.nx = get_scene_features(image, self.normalize_input, margins, self.fsz, segment_image)

    def perform_brl_learning(self):
        """
        Perform Bayesian Reinforcement Learning
        """
        # Make sure that this is only done during training, update_components is used for deployment
        if self.isLearning == False:
            return

        # Predicts action given input state
        pxm_n = np.zeros((1, self.PN + 1))
        for j in range(self.PN + 1):
            a = self.a[0, j] - self.n + 1
            m = self.m[:, j]
            b = (self.a[0, j] - self.n + 1) * self.v[0, j] / (self.v[0, j] + 1) * np.linalg.inv(self.b[:, :, j])
            pxm_n[0, j] = self.calculate_mvtpdf(self.nx, a, m, b, self.n)

        pxm_n = normalize(pxm_n, axis=1)

        pmc_n = self.pmc
        pc_n = self.pc

        pxc_n = pxm_n @ pmc_n
        pxc_n = normalize(pxc_n, axis=1)

        pcx_n = pxc_n.T * pc_n
        pcx_n = normalize(pcx_n, axis=0)

        pmcx_n = np.zeros((self.PN + 1, self.CN))

        for j in range(self.CN):
            pmcx_n[:, j] = normalize(pxm_n.T[:, 0] * pmc_n[:, j])

        # mmac is the component with the greatest effect on selecting Concept
        mmac = self.pmcx[:, self.Cg].argmax()
        cgp = pcx_n.argmax()  # new prediction
        mmacp = pmcx_n[:, cgp].argmax()

        # App1: original TDerr Q-learning off-policy TD control
        self.TDerr = self.Reward + self.forgettingFactor * self.p[cgp, mmacp] - self.p[self.Cg, mmac]

        if self.TDerr > self.posSample:
            self.w = np.expand_dims(self.pmcx[:, self.Cg], 1)

        elif self.TDerr < self.negSample:
            self.w = np.expand_dims(self.pmncx[:, self.Cg], 1)
            self.decisions[-1] *= -1

        else:
            self.w = self.pmx
            self.decisions[-1] *= -0.1

        # Component with minimum distance to the input X will be selected
        all_x = self.x * np.ones((1, self.PN + 1))
        use_linf = True
        if use_linf:
            all_diff = np.abs(self.m[:, 0:self.PN + 1] - all_x)
            all_dis = all_diff.max(0)
        else:
            all_dis = np.linalg.norm(self.m[:, 0:self.PN + 1] - all_x, ord=2)

        ind_dis = all_dis.argmin()
        self.dis = all_dis[ind_dis]
        self.ind_dis = ind_dis
       
        dis_min = 0.2
        dis_max = 0.2

        # ind_dis: Component being updated
        if self.dis < self.disVal or self.TDerr > self.negSample:
            self._update_component(ind_dis)
        else:  # One new component will be added to the network
            self._create_component()
       
    def _create_component(self):

        self.PN += 1
        self.a[:, self.PN] = self.n
        self.v[:, self.PN] = self.n + 1
        self.m[:, self.PN] = self.x[:, 0]
        self.b[:, :, self.PN] = self.initVar * np.eye(self.n)
        self.p[:, self.PN] = np.ones(self.CN) * 0.1
       
        if self.TDerr > self.posSample:
            self.p[self.Cg, self.PN] = 1.

        elif self.TDerr < self.negSample:
            self.p[self.Cg, self.PN] = -1.

    def _update_component(self, idx):

        # Applying weights to Q-function update
        self.p[self.Cg, idx] += self.learningRate_Val * self.w[idx, 0] * self.TDerr

        d = self.x[:, 0] - self.m[:, idx]
        d2 = np.outer(d, d)

        w_1 = self.v[0, idx] * self.w[idx, 0] / (self.v[0, idx] + self.w[idx, 0])
        self.b[:, :, idx] += w_1 * d2

        w_2 = self.w[idx, 0] / (self.v[0, idx] + self.w[idx, 0])
        self.m[:, idx] += w_2 * d

        self.a[0, idx] += self.w[idx, 0]
        self.v[0, idx] += self.w[idx, 0]

    @staticmethod
    def calculate_mvtpdf(X, A, M, B, n):
        """
            Calculates the Probability of X given the multivariate t-distribution
            :param X: State
            :param A: Parameters
            :param M: Parameters
            :param B: Parameters
            :param n: Dimensionality of state
            :return: Probability score
            """
        b = np.exp(gammaln((A + n) / 2.0) - gammaln(A / 2.0)) * np.sqrt(np.linalg.det(B) / ((A * np.pi) ** n))
        d = np.expand_dims(X[:, 0] - M, 1)
        out = b * np.sqrt((1 + d.T @ B @ d / A) ** (-A - n))  # (-A - 2))

        return out[0]

    def calculate_pm_pc_pmc(self):
        """
        Calculates the three conditional probabilities:
        P(M), P(C) and P(M | C)
       """
        p = self.p[:, 0:self.PN + 1]
        mm = np.min(p)
        # os is the bias, it makes p slightly positive by term=abs(mm)/(abs(mm)+1),
        # to produce probs, pmc,pm and pc,
        os = abs(mm) / (abs(mm) + 1) - mm
        # pmc(P(M|C)) is biased and normalized p
        self.pmc = normalize(np.transpose(p + os), axis=1)

        self.pc = (p + os).sum(axis=1, dtype='float')
        self.pc = normalize(self.pc, axis=0)
        self.pc = np.expand_dims(self.pc, 1)

        self.pm = (p + os).sum(axis=0, dtype='float')
        self.pm = normalize(self.pm, axis=0)
        self.pm = np.expand_dims(self.pm, 1)

    def update_components(self):
        """
        Calculate some of the components per agent?
        """

        self.pxm = np.zeros((1, self.PN+1))
        for j in range(self.PN+1):
            a = self.a[0, j]-self.n+1
            m = self.m[:, j]
            b = (self.a[0, j]-self.n+1) * (self.v[0, j]/(self.v[0, j]+1)) * np.linalg.inv(self.b[:, :, j])
            self.pxm[0, j] = self.calculate_mvtpdf(self.x, a, m, b, self.n)

        self.pxm = normalize(self.pxm, axis=1)

        self.calculate_pm_pc_pmc()

        self.pxc = self.pxm @ self.pmc
        self.pxc = normalize(self.pxc, axis=1)

        self.pcx = self.pxc.T * self.pc
        self.pcx = normalize(self.pcx, axis=0)

        self.pmx = self.pxm.T * self.pm
        self.pmx = normalize(self.pmx, axis=0)

        # pmcx: P(M|C, X)
        self.pmcx = np.zeros((self.PN+1, self.CN))
        for j in range(self.CN):
            vec_1 = np.expand_dims(self.pmc[:, j], 1)
            vec_2 = (self.pxm.T * vec_1)/self.pxc[0, j]
            vec_2 = normalize(vec_2, axis=0)
            self.pmcx[:, j] = vec_2[:, 0]

        self.pmncx = np.zeros((self.PN+1, self.CN))
        for j in range(self.CN):
            vec_n1 = np.expand_dims(1.0001-self.pmc[:, j], 1)
            vec_n2 = (self.pxm.T * vec_n1)/self.pxc[0, j]
            vec_n2 = normalize(vec_n2, axis=0)
            self.pmncx[:, j] = vec_n2[:, 0]

    def update_learning_parameters(self):
        if self.isLearning:
            self.disVal = self.disVal + self.disRate * (self.disFinal - self.disVal)
            self.pxmVal = self.pxmVal + self.pxmRate * (self.pxmFinal - self.pxmVal)
            self.pmxVal = self.pmxVal + self.pmxRate * (self.pmxFinal - self.pmxVal)
            self.pcxVal = self.pcxVal + self.pcxRate * (self.pcxFinal - self.pcxVal)
            self.tmpVal = self.tmpVal + self.tmpRate * (self.tmpFinal - self.tmpVal)

    def update_log_statistics(self):
        if self.isLearning:
            self.log_learningRate.append(self.learningRate_Val)
            self.log_cg.append(self.Cg)
            self.log_dist.append(self.disVal)
            self.log_dis.append(self.dis)
            self.log_pxm.append(self.pxmVal)
            self.log_pmx.append(self.pmxVal)
            self.log_pcx.append(self.pcxVal)
            self.log_aveR.append(self.aveR)
            self.log_Reward.append(self.Reward)
            self.log_PN.append(self.PN)
            self.log_TDerr.append(self.TDerr)

    def make_decision(self, method):
        if method == 'Greedy':
            self.Cg = np.argmax(self.probw * self.pcx)
            self.gd += 1

        if method == 'Epsilon_Greedy':
            pcx_e = np.exp(self.pcx)
            pcx_e = pcx_e / pcx_e.sum(axis=0, dtype='float')

            # Greedy Decision using direction prior
            gd = np.argmax(pcx_e) 
            # Epsilon Greedy Decision
            pr = np.ones((self.CN, 1)) * (1. - self.tmpVal) / self.CN
            pr[gd, 0] = pr[gd, 0] + self.tmpVal
            pr = normalize(pr)

            self.Cg = np.random.choice(self.CN, p=pr[:, 0])

            if self.isLearning == 0:
                self.Cg = gd

            if self.Cg == gd:
                self.gd += 1

