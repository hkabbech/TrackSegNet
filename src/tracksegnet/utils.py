"""Other functions

This module implements a extra utils functions.
"""

# Third-party modules
import os

def check_parms(parms_df):
    """Check each parameter value and raise an error if uncorrect.

    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :return: Refined parameters with type.
    :rtype: dict
    """
    if not os.path.isdir(parms_df['data_path']):
        raise ValueError('data_path does not exist.')

    if parms_df['track_format'] != 'MDF' and parms_df['track_format'] != 'CSV':
        raise ValueError('track_format should be MDF or CSV.')

    try:
        parms_df['time_frame'] = float(parms_df['time_frame'])
    except Exception as exc:
        raise ValueError('time_frame should be a floating value.') from exc

    try:
        parms_df['pixel_size'] = float(parms_df['pixel_size'])
    except Exception as exc:
        raise ValueError('pixel_size should be a floating value.') from exc

    try:
        parms_df['num_states'] = int(parms_df['num_states'])
    except Exception as exc:
        raise ValueError('num_states should be an integer between 2 and 6.') from exc
    if parms_df['num_states'] < 2 or parms_df['num_states'] > 6:
        raise ValueError('num_states should be an integer between 2 and 6.')

    for state in range(1, parms_df['num_states']+1):
        if not f'state_{state}_diff' in parms_df:
            raise ValueError(f'state_{state}_diff not defined in parms.csv')
        if not f'state_{state}_alpha' in parms_df:
            raise ValueError(f'state_{state}_alpha not defined in parms.csv')
        try:
            parms_df[f'state_{state}_diff'] = float(parms_df[f'state_{state}_diff'])
        except Exception as exc:
            raise ValueError(f'state_{state}_diff should be a floating value [dimensionless].') from exc

        try:
            parms_df[f'state_{state}_alpha'] = float(parms_df[f'state_{state}_alpha'])
        except Exception as exc:
            raise ValueError(f'state_{state}_alpha should be a floating value between 0 and 2.') from exc
        if parms_df[f'state_{state}_alpha'] < 0 or parms_df[f'state_{state}_alpha'] > 2:
            raise ValueError(f'state_{state}_alpha should be a floating value between 0 and 2.')
        total = 0
        for state2 in range(1, parms_df['num_states']+1):
            if not f'pt_{state}_{state2}' in parms_df:
                raise ValueError(f'pt_{state}_{state2} not defined in parms.csv')
            try:
                parms_df[f'pt_{state}_{state2}'] = float(parms_df[f'pt_{state}_{state2}'])
            except Exception as exc:
                raise ValueError(f'pt_{state}_{state2} should be a floating value.') from exc
            total += parms_df[f'pt_{state}_{state2}']
        if total != 1:
            raise ValueError(f'Total probability in state {state} should equal 1.')
    return parms_df

def get_color_list(num_states):
    """Create list of colors.
    
    :param num_states: Number of states.
    :type num_states: int
    :return: List of colors for the plots.
    :rtype: list
    """
    if num_states == 1:
        print('Please, select 2 to 6 states.')
    if num_states == 2:
        colors = ['darkblue', 'red']
    elif num_states == 3:
        colors = ['darkblue', 'darkorange', 'red']
    elif num_states == 4:
        colors = ['darkblue', 'darkorange', 'red', 'darkviolet']
    elif num_states == 5:
        colors = ['darkblue', 'darkorange', 'red', 'darkviolet', 'green']
    elif num_states == 6:
        colors = ['darkblue', 'darkorange', 'red', 'darkviolet', 'green', 'k']
    return colors
