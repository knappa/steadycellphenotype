import json
from pathlib import Path
import tempfile
from typing import Set

from flask import make_response, Markup, Response
import matplotlib
import matplotlib.pyplot as plt

from steady_cell_phenotype._util import *

matplotlib.use('agg')


def get_trajectory_edge_list(*,
                             init_state: np.ndarray,
                             update_fn: Callable,
                             target_variables: Tuple[str]) -> List[Dict[str, Dict]]:
    phased_trajectory: np.ndarray
    phased_trajectory, phase_point = get_trajectory(init_state=init_state, update_fn=update_fn)

    edge_list = [{'source': dict(zip(target_variables,
                                     phased_trajectory[i, :])),
                  'target': dict(zip(target_variables,
                                     phased_trajectory[i + 1, :]))}
                 for i in range(phased_trajectory.shape[0] - 1)]
    return edge_list


def add_nearby_states(init_states: List[np.ndarray]) -> List[np.ndarray]:
    """
    Adds all states Hamming distance 1 from an init state to the list
    Parameters
    ----------
    init_states

    Returns
    -------
    init_states plus hamming distance one states. Order not preserved.
    """
    if len(init_states) <= 0:
        return []

    new_init_states: Set[HashableNdArray] = set()
    num_vars: int = init_states[0].shape[0]

    # add all Hamming distance `dist` states from the initial states
    for state in init_states:
        for idx in range(num_vars):
            for val in range(3):
                if abs(state[idx] - val) <= 1:
                    new_state = state.copy()
                    new_state[idx] = val
                    new_init_states.add(HashableNdArray(new_state))

    return [state.array for state in new_init_states]


def run_model_variable_initial_values(*,
                                      init_state_prototype: Dict[str, str],
                                      variables: List[str],
                                      update_fn: Callable,
                                      equation_system: EquationSystem,
                                      check_nearby: bool) -> List:
    """
    Run the model on possibly *'ed sets of initial conditions

    Parameters
    ----------
    init_state_prototype
        A dictionary of variable names and their initial values {0,1,2,*} where * is a wildcard
    variables
        The variables of the model, in order
    update_fn
    equation_system
    check_nearby
        If true, we include states Hamming distance 1 from the initial state(s).

    Returns
    -------
    List of edges
    """

    # expand any *'ed variables
    def get_states(initial_state: Dict[str, str],
                   remaining_variable_states: List[str]) -> List[np.ndarray]:
        if len(remaining_variable_states) == 0:
            return [np.array([int(initial_state[var]) for var in variables], dtype=np.int64)]
        else:
            initial_states = []
            for val in range(3):
                resolved_state = initial_state.copy()
                resolved_state[remaining_variable_states[0]] = str(val)
                initial_states += get_states(resolved_state, remaining_variable_states[1:])
            return initial_states

    variable_states = [k for k in init_state_prototype if init_state_prototype[k] == '*']
    try:
        init_states: List[np.ndarray] = get_states(init_state_prototype, variable_states)
    except ValueError:
        raise Exception(make_response(error_report(
                "Error constructing initial states"
                )))

    # check to see if we will be overloaded
    if len(variable_states) > MAX_SUPPORTED_VARIABLE_STATES:
        raise Exception(make_response(error_report(
                f"Web platform is limited to {MAX_SUPPORTED_VARIABLE_STATES} variable states.")))

    if check_nearby:
        init_states = add_nearby_states(init_states)

    edge_lists = []
    for state in init_states:
        edge_lists.append(get_trajectory_edge_list(init_state=state,
                                                   update_fn=update_fn,
                                                   target_variables=equation_system.target_variables()))
    return edge_lists


def compute_trace(*,
                  model_text: str,
                  knockouts: Dict[str, str],
                  continuous: Dict[str, bool],
                  init_state: Dict[str, str],
                  visualize_variables: Dict[str, bool],
                  check_nearby: bool):
    """ Run the cycle finding simulation for an initial state """

    # create an update function and equation system
    variables, update_fn, equation_system = process_model_text(model_text, knockouts, continuous)

    # construction of initial values can fail with an exception that contains a response
    try:
        edge_lists = run_model_variable_initial_values(init_state_prototype=init_state,
                                                       variables=variables,
                                                       update_fn=update_fn,
                                                       equation_system=equation_system,
                                                       check_nearby=check_nearby)
    except Exception as e:
        payload = e.args[0]
        if type(payload) in {str, Response}:
            return payload
        else:
            # should not occur
            return make_response(error_report("Unknown error"))

    def to_key(edges):
        return frozenset(edges.items())

    # give numeric labels to vertices
    labels = dict()
    count = 0
    for edge_list in edge_lists:
        for edge in edge_list:
            source = to_key(edge['source'])
            if source not in labels:
                labels[source] = count
                count += 1
            target = to_key(edge['target'])
            if target not in labels:
                labels[target] = count
                count += 1

    return_states = []
    for edge_list in edge_lists:
        return_state = to_key(edge_list[-1]['target'])
        return_states.append(labels[return_state])

    source_labels = [[labels[to_key(edge['source'])] for edge in edge_list] for edge_list in edge_lists]

    variable_level_plots = plot_variable_levels_for_trajectories(edge_lists=edge_lists,
                                                                 return_states=return_states,
                                                                 source_labels=source_labels,
                                                                 variables=variables,
                                                                 visualize_variables=visualize_variables)

    # create data for the javascript
    nodes_json = json.dumps([{'id': label} for label in labels.values()])

    edge_tuples = set()
    for edge_list in edge_lists:
        edge_tuples.update({(labels[to_key(edge['source'])], labels[to_key(edge['target'])])
                            for edge in edge_list})
    edge_json = json.dumps([{'source': source, 'target': target} for (source, target) in edge_tuples])

    # respond with the results-of-computation page
    return make_response(render_template('compute-trace.html',
                                         variables=equation_system.target_variables(),
                                         num_edge_lists=len(edge_lists),
                                         trajectories=list(zip(edge_lists,
                                                               return_states,
                                                               source_labels,
                                                               map(len, edge_lists),
                                                               variable_level_plots)),
                                         nodes=nodes_json,
                                         links=edge_json))


def plot_variable_levels_for_trajectories(*,
                                          edge_lists,
                                          return_states,
                                          source_labels,
                                          variables: List[str],
                                          visualize_variables: Dict[str, bool]):
    # mask for which variables to viz
    visualize_variable_mask = np.array([visualize_variables[var] for var in variables], dtype=bool)

    if np.sum(visualize_variable_mask) <= 0:
        # no viz requested
        return [''] * len(edge_lists)

    # draw a visualization of variable levels
    variable_level_plots = []
    for n, edge_list in enumerate(edge_lists):
        plt.rcParams['svg.fonttype'] = 'none'
        plt.figure(figsize=(6, 2))

        # prepare to plot
        data = np.array([
            [edge['source'][variable] for edge in edge_list] + [edge_list[-1]['target'][variable]]
            for variable in variables]).T

        # plot
        plt.plot(data[:, visualize_variable_mask])

        # note the repeat region
        if return_states[n] in source_labels[n]:
            plt.axvline(source_labels[n].index(return_states[n]), color='k', linestyle='dotted')
        plt.axvline(len(edge_list), color='k', linestyle='dotted')

        # limits and ticks
        plt.ylim([-0.25, 2.25])
        plt.yticks([0, 1, 2])
        plt.xlim([-0.25, len(edge_list) + 0.25])
        plt.xticks(np.arange(len(edge_list) + 1))
        plt.gca().set_xticklabels(source_labels[n] + [return_states[n]])

        # legend and labels
        plt.legend(np.array(variables)[visualize_variable_mask],
                   bbox_to_anchor=(1.04, 1),
                   loc="center left")
        plt.ylabel("Level")
        plt.xlabel("State")
        plt.tight_layout()
        image_filename = f'levels{n}.svg'
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file = Path(tmp_dir_name) / image_filename
            plt.savefig(tmp_file, transparent=True, pad_inches=0.0)
            plt.close()
            with open(tmp_file, 'r') as image:
                variable_level_plots.append(Markup(image.read()))
    return variable_level_plots
