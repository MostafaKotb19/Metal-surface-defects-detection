import argparse
import omegaconf.dictconfig

def dict2namespace(config):
    """
    Converts a dictionary to a namespace.

    Parameters:
        config (dict): The dictionary to be converted.

    Returns:
        argparse.Namespace: The namespace containing the dictionary values.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def extract_min(lst, index):
    """
    Extracts the minimum value from a nested list based on the specified index.
    
    Args:
        lst (list): A nested list containing numerical values.
        index (int): The index to compare the values.
        
    Returns:
        int: The minimum value found in the nested list.
    """
    minn = 10000000
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j][index] < minn:
                minn = lst[i][j][index]
    return minn

def extract_max(lst, index):
    """
    Extracts the maximum value from a nested list based on the given index.
    
    Args:
        lst (list): A nested list containing numerical values.
        index (int): The index to compare the values.
        
    Returns:
        int: The maximum value found in the nested list based on the given index.
    """
    maxx = -10000000
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j][index] > maxx:
                maxx = lst[i][j][index]
    return maxx

def max_min_scaler(lst, maxx, minn, index):
    """
    Applies max-min scaling to a list of values.

    Parameters:
    lst (list): The list of values to be scaled.
    maxx (float): The maximum value for scaling.
    minn (float): The minimum value for scaling.
    index (int): The index of the value to be scaled within each sublist.

    Returns:
    list: The scaled list of values.
    """
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j][index] != 0:
                lst[i][j][index] = (lst[i][j][index] - minn) / (maxx - minn)
    return lst