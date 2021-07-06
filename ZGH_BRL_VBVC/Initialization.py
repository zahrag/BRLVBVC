
from ZGH_BRL_VBVC.BRLearner import BRLearner


class InitInfo:
    """Common base class for required initial information"""

    def __init__(self, Learner='FOA', Param=-0.01, episode=0, phase=0, epi_max_I=2, epi_max_II=4500, epi_max_III=4500,
                 MotorAction=['fast_forward', 'right_turn', 'left_turn', 'slow_down', 'reverse', 'stop']
                 ):

        # Decide if print should be done
        self.verbose = False

        self.Param = Param
        self.MotorAction = MotorAction
        self.Learner = Learner
        self.episode = episode
        self.phase = phase
        self.epi_max_I = epi_max_I
        self.epi_max_II = epi_max_II
        self.epi_max_III = epi_max_III
        self.shutdown = False
        self.was_reset = False
        self.acc_collision_ped = 0.
        self.acc_collision_veh = 0.
        self.acc_collision_oth = 0.
        self.control_hist = {
            'master': [0, 0, 0, 0, 0, 0, 0],
            'motor': [0, 0, 0, 0, 0, 0]
        }
        self.m_act_no = 4

    def reset_collision(self):
        """
        Resets the accumulated collision
        """
        self.acc_collision_ped = 0.
        self.acc_collision_veh = 0.
        self.acc_collision_oth = 0.


def init_all():

    """
        This would create the object of the class
        """
    info = InitInfo()
    all_BRLearner = []
    all_BRLearner.append(init_Agent(info.m_act_no))

    return info, all_BRLearner


def init_Agent(m_act_no):

    """
        This function creates a Full Observer Agent (FOA).
        """
    n_grid = 6
    f_size = 5
    BRL = BRLearner(type='FOA',            isLearning=False,        normalize_input=True, depth_dist=False,
                    idx=1,                 fsz=f_size,              CN=m_act_no,          num_grids=n_grid,
                    forgettingFactor=0.8,  posSample=-5,            negSample=-10,
                    learningRate_Val=0.99, learningRate_Rate=1e-5,  learningRate_Final=0.01,
                    tmpVal=0.5,            tmpRate=7e-3,            tmpFinal=0.99,
                    disVal=0.1,            disRate=3e-5,            disFinal=0.01,
                    pxmVal=0.5,            pxmRate=3e-1,            pxmFinal=20,
                    pmxVal=1.1,            pmxRate=1e-2,            pmxFinal=0.2,
                    pxcVal=0.2,            pxcRate=0.01,            pxcFinal=0.2,
                    )

    return BRL


