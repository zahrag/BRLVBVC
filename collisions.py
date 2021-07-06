import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from scipy.special import gamma as scigamma
from scipy.special import gammaln as scigammaln
from collections import OrderedDict


def split_into_tasks(reader):
    out = list()
    tmp = OrderedDict()
    last_weather = -1
    last_start = -1
    last_end = -1
    for row in reader:
        weather = row['weather']
        startp = row['start_point']
        endp = row['end_point']
        if weather != last_weather or startp != last_start or endp != last_end:
            # Add new task
            if tmp:
                out.append(tmp)
                tmp = OrderedDict()
            last_weather = weather
            last_start = startp
            last_end = endp
        if tmp:
            for key in row:
                tmp[key].append(float(row[key]))
        else:
            for key in row:
                tmp[key] = [float(row[key])]
#        tmp.update(row)

    if tmp:
        out.append(tmp)

    return out


def get_total_distance_of_tasks(task_data):
    distance = list()
    for task in task_data:
        x_diff = np.diff(task['pos_x'])
        y_diff = np.diff(task['pos_y'])

        acc_dist = np.cumsum(np.sqrt(x_diff ** 2 + y_diff ** 2))
        distance.append(acc_dist[-1])

    return np.asarray(distance)


def get_successful_tasks(task_data, key):
    successes = list()
    for task in task_data:
        data = np.asarray(task[key])
        if np.any(data > 0):
            successes.append(False)
        else:
            successes.append(True)

    return np.asarray(successes)


def get_distance_between_infractions(task_data, key):
    distances = list()
    acc_dist = 0
    for task in task_data:
        data = np.array(task[key])
        x_diff = np.diff(task['pos_x'])
        y_diff = np.diff(task['pos_y'])

        distance = np.cumsum(np.sqrt(x_diff ** 2 + y_diff ** 2))
        if not np.any(data > 0):
            # Accumulate distance between tasks
            acc_dist += distance[-1]
            continue

        indices = np.flatnonzero(data)
        distances.append(acc_dist + distance[indices[0] - 1])
        # Reset accumulated distance for next infraction
        acc_dist = 0

    if not distances:
        distances.append(acc_dist)

    return np.asarray(distances)


def get_distance_to_first_infraction(task_data, key):
    distances = list()
    for task in task_data:
        data = np.array(task[key])
        if not np.any(data > 0):
            continue
        x_diff = np.diff(task['pos_x'])
        y_diff = np.diff(task['pos_y'])

        distance = np.cumsum(np.sqrt(x_diff**2 + y_diff**2))
        indices = np.flatnonzero(data)

        distances.append(distance[indices[0]-1])

    return np.asarray(distances)


def get_percentage_under_infraction(task_data, key, threshold):
    percentages = list()
    if type(key) != list:
        key = [key]
    for task in task_data:
        data = list()
        for k in key:
            data.append(np.asarray(task[k]))
        data = np.asarray(data)
        infractions = data > threshold
        infractions = np.logical_or.reduce(infractions)
        percentages.append(np.sum(infractions) / data.shape[1])
    return np.asarray(percentages)


def get_hist_of_infractions(task_data, nbins=100):
    data_offroad = []
    data_otherside = []
    for task in task_data:
        data_offroad += task['intersection_offroad']
        data_otherside += task['intersection_otherlane']

    data_offroad = np.asarray(data_offroad)
    data_otherside = np.asarray(data_otherside)
    data = np.concatenate((data_offroad, -data_otherside))
    hist = np.histogram(data, bins=nbins, range=(-1, 1))

    return data


def estimate_binomial_distribution(success_statuses):
    s = np.sum(success_statuses)
    f = len(success_statuses) - s
    return s, f


def estimate_beta_distribution(distances):
    dist = np.array(distances, dtype=np.float)
    x = np.mean(dist)
    v = np.var(dist)

    alpha = x * (x*(1-x) / v - 1)
    beta = (1 - x) * (x*(1-x) / v - 1)

    return alpha, beta



def get_pdf_beta_posterior(s, f, prior='jeffreys'):
    x = np.linspace(0.05, 0.95, 100)
    if prior == 'bayes':
        pdf = (x**s * (1-x)**f) * scigamma(s + f + 2) / (scigamma(s + 1) * scigamma(f + 1))
    elif prior == 'jeffreys':
        pdf = (x ** (s-0.5) * (1 - x) ** (f-0.5)) * scigamma(s + f + 1) / (scigamma(s + 0.5) * scigamma(f + 0.5))
    return pdf, x


def get_pdf_beta_posterior_in_logarithm(s, f, prior='jeffreys'):
    x = np.linspace(0.001, 0.999, 1000)
    lx= np.log(x)
    lnx = np.log(1-x)
    if prior == 'jeffreys':
        lpdf = (s-0.5)*lx + (f-0.5)*lnx + scigammaln(s + f + 1) - scigammaln(s + 0.5) - scigammaln(f + 0.5)
    return np.exp(lpdf), x


def get_pdf_for_infractions(filename):

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Split measurments into tasks
        tasks = split_into_tasks(reader)


        infr_dists_other = get_distance_between_infractions(tasks, 'collision_other')
        infr_dists_pedestrian = get_distance_between_infractions(tasks, 'collision_pedestrians')
        infr_dists_vehicle = get_distance_between_infractions(tasks, 'collision_vehicles')

        # Estimate for accidents
        gamma_distr = list()

        # Estimate for accidents
        exp_distr = list()

        # Estimate for percentage of non-collision
        successes_other = get_successful_tasks(tasks, 'collision_other')
        successes_pedestrians = get_successful_tasks(tasks, 'collision_pedestrians')
        successes_vehicles = get_successful_tasks(tasks, 'collision_vehicles')

        beta_distr = list()
        s, f = estimate_binomial_distribution(successes_other)
        pdf, x = get_pdf_beta_posterior_in_logarithm(s, f)
        coll_free = {'s': s, 'f': f}

        beta_distr.append({'pdf': pdf, 'params': (s, f)})
        s, f = estimate_binomial_distribution(successes_pedestrians)
        pdf, x = get_pdf_beta_posterior_in_logarithm(s, f)
        beta_distr.append({'pdf': pdf, 'params': (s, f)})
        s, f = estimate_binomial_distribution(successes_vehicles)
        pdf, x = get_pdf_beta_posterior_in_logarithm(s, f)
        beta_distr.append({'pdf': pdf, 'params': (s, f)})

        beta_x = x

    return exp_distr, gamma_distr, beta_distr, tasks, exp_x, gamma_x, beta_x, coll_free


def get_success_status(filename):
    success = list()
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            reached_goal = int(line['result'])
            success.append(reached_goal == 1)

    return success


def calculate_KL_divergence_exponential(lambda1, lambda2):
    """
    Calculates distance from true distribution to approximate distribution
    Args:
        lambda1: "True" distribution
        lambda2: "Approximate" distribution

    Returns:
        KL divergence
    """
    return np.log(lambda1) - np.log(lambda2) + lambda2 / lambda1 - 1


def sample_from_exponential_bayesian(data, a=1, b=1, nsamples=100):
    dist = np.asarray(data)
    xbar = np.mean(dist)
    n = len(dist)

    ls = np.random.gamma(a+n, 1/(b+n*xbar), nsamples)
    for l in ls:
        cdf, x = get_cdf_exponential(l)
        plt.plot(x, cdf)
    plt.show()


if __name__ == '__main__':
    #np.seterr(all='raise')

    # Metrics for GT models
    exp_distr = {}
    gamma_distr = {}
    beta_distr = {}
    task_data = {}
    exp_x = {}
    gamma_x = {}
    beta_x = {}
    results = {}
    beta_success = {}
    beta_success_x = {}
    folder_names = {}
    beta_success_acc = {}
    beta_coll_acc = {}
    keys = ['setting']
    for k in keys:
        exp_distr[k] = list()
        gamma_distr[k] = list()
        beta_distr[k] = list()
        task_data[k] = list()
        exp_x[k] = list()
        gamma_x[k] = list()
        beta_x[k] = list()
        results[k] = list()
        beta_success[k] = list()
        beta_success_x[k] = list()
        folder_names[k] = list()
        beta_success_acc[k] = list()
        beta_coll_acc[k] = list()


    base_name = {'setting': 'path/to/carla_results_folder'}
    nums = {'setting': None}


    for key in base_name:
        if nums[key] is not None:
            for n in nums[key]:
                folder_names[key].append(base_name[key] + '{}'.format(n))
        else:
            folder_names[key].append(base_name[key])


    for key in folder_names:
        s_acc_goal = 0
        f_acc_goal = 0
        s_acc_coll = 0
        f_acc_coll = 0
        for folder in folder_names[key]:
            exp_distr1, gamma_distr1, beta_distr1, task_data1, exp_x1, gamma_x1, beta_x1, coll_free = get_pdf_for_infractions(
                folder + '/measurements.csv')
            reached_goal = get_success_status(folder + '/summary.csv')
            exp_distr[key].append(exp_distr1)
            gamma_distr[key].append(gamma_distr1)
            beta_distr[key].append(beta_distr1)
            task_data[key].append(task_data1)
            exp_x[key].append(exp_x1)
            gamma_x[key].append(gamma_x1)
            beta_x[key].append(beta_x1)
            results[key].append(reached_goal)
            s, f = estimate_binomial_distribution(reached_goal)
            #pdf, x = get_pdf_beta_posterior(s, f)
            pdf, x = get_pdf_beta_posterior_in_logarithm(s, f)
            beta_success[key].append(pdf)
            beta_success_x[key].append(x)
            s_acc_goal += s
            f_acc_goal += f
            s_acc_coll += coll_free['s']
            f_acc_coll += coll_free['f']

        if len(folder_names[key]) > 1:
            pdf, x = get_pdf_beta_posterior_in_logarithm(s_acc_goal, f_acc_goal)
            beta_success_acc[key] = {'pdf': pdf, 'x': x}

            pdf, x = get_pdf_beta_posterior_in_logarithm(s_acc_coll, f_acc_coll)
            beta_coll_acc[key] = {'pdf': pdf, 'x': x}


    # Estimate a ranking of the models
    offroad_percentage = {}
    otherlane_percentage = {}
    combined_percentage = {}
    distance = {}
    accomplished_tasks = {}
    collisionfree_tasks = {}
    score = {}
    best_idx = {}
    worst_idx = {}
    median_idx = {}

    for k in keys:
        offroad_percentage[k] = {"mean": -1, "median": -1, "best": -1, "worst": -1, "all": list()}
        otherlane_percentage[k] = {"mean": -1, "median": -1, "best": -1, "worst": -1, "all": list()}
        combined_percentage[k] = {"mean": -1, "median": -1, "best": -1, "worst": -1, "all": list()}
        distance[k] = {"mean": -1, "median": -1, "best": -1, "worst": -1, "all": list()}
        accomplished_tasks[k] = {"mean": -1, "median": -1, "best": -1, "worst": -1, "all": list()}
        collisionfree_tasks[k] = {"mean": -1, "median": -1, "best": -1, "worst": -1, "all": list()}
        score[k] = list()
        best_idx[k] = -1
        worst_idx[k] = -1
        median_idx[k] = -1

    for key in keys:
        total_dist = 0.
        percentage_for_infraction = 0.2
        for i, (task, beta_d, reached_goal) in enumerate(zip(task_data[key], beta_distr[key], results[key])):
            offroad_percentage[key]["all"].append(100. * np.mean(get_percentage_under_infraction(task, 'intersection_offroad',
                                                                 percentage_for_infraction)))
            otherlane_percentage[key]["all"].append(100. * np.mean(get_percentage_under_infraction(task, 'intersection_otherlane',
                                                                   percentage_for_infraction)))
            combined_percentage[key]["all"].append(100. * np.mean(get_percentage_under_infraction(task,
                                                                  ['intersection_offroad', 'intersection_otherlane'],
                                                                  percentage_for_infraction)))

            distance[key]["all"].append(np.sum(get_total_distance_of_tasks(task)))

            s, fails = beta_d[0]['params']
            collisionfree_tasks[key]["all"].append( 100. * s / (s + fails) )
            res = np.sum(reached_goal)
            accomplished_tasks[key]["all"].append(100 * res / len(reached_goal))

            score[key].append((100. - combined_percentage[key]["all"][-1] + collisionfree_tasks[key]["all"][-1]
                         + accomplished_tasks[key]["all"][-1])/300)

        scores = np.asarray(score[key])
        best_idx[key] = np.argmax(scores)
        worst_idx[key] = np.argmin(scores)
        median_idx[key] = np.argmax(np.median(scores) == scores)

        offroad_percentage[key]["best"] = offroad_percentage[key]["all"][best_idx[key]]
        offroad_percentage[key]["worst"] = offroad_percentage[key]["all"][worst_idx[key]]
        offroad_percentage[key]["median"] = np.median(offroad_percentage[key]["all"])  #[median_idx[key]]
        offroad_percentage[key]["mean"] = np.mean(offroad_percentage[key]["all"])

        otherlane_percentage[key]["best"] = otherlane_percentage[key]["all"][best_idx[key]]
        otherlane_percentage[key]["worst"] = otherlane_percentage[key]["all"][worst_idx[key]]
        otherlane_percentage[key]["median"] = np.median(otherlane_percentage[key]["all"])  # [median_idx[key]]
        otherlane_percentage[key]["mean"] = np.mean(otherlane_percentage[key]["all"])

        combined_percentage[key]["best"] = combined_percentage[key]["all"][best_idx[key]]
        combined_percentage[key]["worst"] = combined_percentage[key]["all"][worst_idx[key]]
        combined_percentage[key]["median"] = np.median(combined_percentage[key]["all"])  # [median_idx[key]]
        combined_percentage[key]["mean"] = np.mean(combined_percentage[key]["all"])

        accomplished_tasks[key]["best"] = accomplished_tasks[key]["all"][best_idx[key]]
        accomplished_tasks[key]["worst"] = accomplished_tasks[key]["all"][worst_idx[key]]
        accomplished_tasks[key]["median"] = np.median(accomplished_tasks[key]["all"])  # [median_idx[key]]
        accomplished_tasks[key]["mean"] = np.mean(accomplished_tasks[key]["all"])

        collisionfree_tasks[key]["best"] = collisionfree_tasks[key]["all"][best_idx[key]]
        collisionfree_tasks[key]["worst"] = collisionfree_tasks[key]["all"][worst_idx[key]]
        collisionfree_tasks[key]["median"] = np.median(collisionfree_tasks[key]["all"])  # [median_idx[key]]
        collisionfree_tasks[key]["mean"] = np.mean(collisionfree_tasks[key]["all"])

        distance[key]["best"] = distance[key]["all"][best_idx[key]]
        distance[key]["worst"] = distance[key]["all"][worst_idx[key]]
        distance[key]["median"] = np.median(distance[key]["all"])  # [median_idx[key]]
        distance[key]["mean"] = np.mean(distance[key]["all"])

    colours = ['cyan', 'brown', 'purple', 'red', 'blue', 'green']
    for i, key in enumerate(keys):
        plt.plot(beta_x[key][best_idx[key]], beta_distr[key][best_idx[key]][0]['pdf'], label=key, lw=3, color=colours[i])

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='T*D*')
    purple_patch = mpatches.Patch(color='blue', label='RL')
    brown_path = mpatches.Patch(color='green', label='IL')

    plt.legend(handles=[red_patch, purple_patch, brown_path],
               loc='upper left', fontsize=24)
    plt.ylim([0, 30])
    plt.xlim([0, 1])
    plt.yticks([])
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['0\%', '25\%', '50\%', '75\%', '100\%'], fontsize=28)
    plt.ylabel(r'$P(p_{\neg Collision} = x\%)$', fontsize=30)
    plt.show()

    for i, key in enumerate(keys):
        plt.plot(beta_success_x[key][best_idx[key]], beta_success[key][best_idx[key]], label=key, lw=3, color=colours[i])

    plt.legend(handles=[red_patch, purple_patch, brown_path],
               loc='upper left', fontsize=24)
    plt.ylim([0, 30])
    plt.xlim([0, 1])
    plt.yticks([])
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['0\%', '25\%', '50\%', '75\%', '100\%'], fontsize=28)
    plt.ylabel(r'$P(p_{Success} = x\%)$', fontsize=30)
    plt.subplots_adjust(hspace=0.05, wspace=0)
    plt.show()

    # Plot the success rate for all models in a specific training condition

    blue_patch = mpatches.Patch(color='blue', label='TGDG')
    orange_patch = mpatches.Patch(color='orange', label='TGDE')
    green_patch = mpatches.Patch(color='green', label='TEDE')
    red_patch = mpatches.Patch(color='red', label='TEDG')

    for i, key in enumerate(keys):
        if beta_success_acc[key]:
            plt.plot(beta_success_acc[key]['x'], beta_success_acc[key]['pdf'], label=key, lw=3)

    plt.legend(handles=[blue_patch, orange_patch, green_patch, red_patch],
               loc='upper left', fontsize=20)
    plt.ylim([0, 45])
    plt.xlim([0, 1])
    plt.yticks([])
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['0\%', '25\%', '50\%', '75\%', '100\%'], fontsize=28)
    plt.ylabel(r'$P(p_{Success} = x\%)$', fontsize=30)
    plt.savefig('results/betaSuccessTot.pdf', bbox_inches='tight')
    plt.show()

    # Plot the success rate for all models in a specific training condition
    for i, key in enumerate(keys):
        if beta_coll_acc[key]:
            plt.plot(beta_coll_acc[key]['x'], beta_coll_acc[key]['pdf'], label=key, lw=3)

    plt.legend(handles=[blue_patch, orange_patch, green_patch, red_patch],
              loc='upper left', fontsize=20)
    plt.ylim([0, 30])
    plt.xlim([0, 1])
    plt.yticks([])
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['0\%', '25\%', '50\%', '75\%', '100\%'], fontsize=28)
    plt.ylabel(r'$P(p_{\neg Collision} = x\%)$', fontsize=30)
    plt.savefig('results/betaCollTot.pdf', bbox_inches='tight')
    plt.show()


    # Plotting histogram for out-of-road
    fig = plt.figure()
    nfigs = len(task_data)
    nbins = 13
    for j, key in enumerate(keys):
        nmodels = 1  # len(task_data[key])
        td = task_data[key][best_idx[key]]
        plt.subplot(nfigs, 1, j+1)
        hist_data = get_hist_of_infractions(td)
        plt.hist2d(hist_data, np.zeros_like(hist_data), bins=[nbins, 1], range=[[-1, 1], [0, 0]], normed=True)
        plt.yticks([])  #np.arange(0, nmodels), fontsize=14)
        plt.ylabel(key, fontsize=14)
        plt.ylim([-0.5, nmodels - 0.5])
        plt.xticks([])

        ax = plt.gca()
        ax.set_xticks(np.linspace(-1, 1, nbins+1), minor=True)
        ax.set_yticks(np.arange(-0.5, nmodels - 0.5, 1), minor=True)
        plt.grid(which='minor', lw=0.5, c='k')
    plt.xticks([-1., -0.5, 0., 0.5, 1.],
               ['100\%', '50\%', '0\%', '50\%', '100\%'], fontsize=28)
    fig.subplots_adjust(hspace=0.05, wspace=0)
    plt.show()

    # Plotting histogram for out-of-road for each model set
    fig = plt.figure()
    nfigs = len(task_data)
    nbins = 13
    for j, key in enumerate(keys):
        nmodels = 1  # len(task_data[key])
        hist_data = np.zeros(nbins)
        for td in task_data[key]:
            hist_data = np.concatenate((hist_data,get_hist_of_infractions(td)))
        plt.subplot(nfigs, 1, j + 1)
        plt.hist2d(hist_data, np.zeros_like(hist_data), bins=[nbins, 1], range=[[-1, 1], [0, 0]], normed=True)
        plt.yticks([])
        plt.ylabel(key, fontsize=14)
        plt.ylim([-0.5, nmodels - 0.5])
        plt.xticks([])

        ax = plt.gca()
        ax.set_xticks(np.linspace(-1, 1, nbins + 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nmodels - 0.5, 1), minor=True)
        plt.grid(which='minor', lw=0.5, c='k')
    plt.xticks([-1., -0.5, 0., 0.5, 1.],
               ['100\%', '50\%', '0\%', '50\%', '100\%'], fontsize=28)
    fig.subplots_adjust(hspace=0.05, wspace=0)
    plt.show()

