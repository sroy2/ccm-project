#!/usr/bin/env python
# -*- coding: utf-8 -*-

def path():
    '''Helper function to get data path within project.
    
    Returns
    -------
    path_to_data : string
    '''
    from pathlib import Path
    
    path = Path('.').resolve()
    path_string = path.absolute().as_posix()
    if 'src' in path_string:
        path = path.parent / 'data'
    elif 'data' in path_string:
        pass
    else:
        path = path / 'data'
    path_to_data = f'{path.absolute().as_posix()}/'
    return path_to_data
    
def data(file='masked_data.txt'):
    '''Helper function to load data.
    
    Returns
    -------
    lines: list
    '''
    location = path()
    lines = []
    with open(f'{location}file) as f:
        for line in f:
            lines.append(line)
    return lines