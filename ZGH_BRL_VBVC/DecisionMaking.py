
def decision_making(agents, directions):
    """
    Let make a decision
    Actions: Forward, Right, Left, Backward
    :param agents: All agents
    :param directions: ['Reach goal', 'Unknown', 'Lane follow', 'Left', 'Right', 'Forward']
    :return: Action and Action type
    """
    a = agents[0]
    a.make_decision('Epsilon_Greedy')
    action = a.Cg
    action_type = 'motor'
    a.log_cg.append([action, action])

    return action, action_type


def action_selection(agents, directions):

    action, action_type = decision_making(agents, directions)

    return agents, action, action_type



















