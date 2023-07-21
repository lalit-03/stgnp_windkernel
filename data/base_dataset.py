"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random

import torch
import torch.utils.data as data
from abc import ABC, abstractmethod
import numpy as np
import os
import time
from math import radians, cos, sin, asin, sqrt
from data.data_util import *

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        In this function, we instantiate the following variables:
        raw_data -- a dictionary that stores the raw data:
            'pred' -- target variables of shape (num_nodes, num_time_steps, num_features1)
            'feat' (optional) -- covariates of shape (num_nodes, num_time_steps, num_features2)
            'time' -- time stamps of shape (num_time_steps, )
            'missing' -- missing masks of shape (num_nodes, num_time_steps, num_features1)
        A -- adjacency matrix of shape (num_nodes, num_nodes)
        test_node_index -- a numpy array of shape (num_test_nodes, ) that stores the indices of test nodes
        train_node_index -- a numpy array of shape (num_train_nodes, ) that stores the indices of train nodes
        """
        self.opt = opt
        self.time_division = {
            'train': [0.0, 0.7],
            'val': [0.8, 0.9],
            'test': [0.9, 1.0]}
        self.raw_data = {}
        self.A = None
        self.test_node_index = None
        self.train_node_index = None

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.phase == 'train':
            length = self.raw_data['feat'].shape[1] - self.opt.t_len
        else:
            length = int(self.raw_data['feat'].shape[1] / self.opt.t_len)
        return length

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """

        num_train_target = self.opt.num_train_target if self.opt.phase == 'train' else None
        batch_data = self._fetch_divided_form_data_item(self.raw_data, self.A, index, self.opt.t_len,
                                                        self.train_node_index, self.test_node_index,
                                                        num_train_target, self.opt.phase)
        return batch_data

    def add_norm_info(self, mean, scale):
        """Add the normalization information of the dataset.
        """
        self.opt.__dict__.update({'mean': mean, 'scale': scale})

    ##################################################
    # utility functions
    ##################################################
    def _data_format_check(self):
        # check raw_data
        if 'pred' not in self.raw_data.keys():
            raise ValueError('raw_data must contain key \'pred\'')
        if 'missing' not in self.raw_data.keys():
            raise ValueError('raw_data must contain key \'missing\'')
        if 'time' not in self.raw_data.keys():
            raise ValueError('raw_data must contain key \'time\'')
        if self.raw_data['pred'].shape != self.raw_data['missing'].shape:
            raise ValueError('pred and missing must have the same shape')
        if 'feat' in self.raw_data.keys():
            if self.raw_data['pred'].shape[:-1] != self.raw_data['feat'].shape[:-1]:
                raise ValueError('pred and feat must have the same shape except the last dimension')
        if not isinstance(self.test_node_index, np.ndarray) or not isinstance(self.train_node_index, np.ndarray):
            raise ValueError('test_node_index and train_node_index must be numpy arrays')
        if len(self.test_node_index.shape) != 1 or len(self.train_node_index.shape) != 1:
            raise ValueError('test_node_index and train_node_index must be 1D arrays')
        # check adjacency matrix
        if not isinstance(self.A, np.ndarray):
            raise ValueError('A must be a numpy array')
        if len(self.A.shape) != 2:
            raise ValueError('A must be a 2D array')
        # norm info check
        if self.opt.mean is None or self.opt.scale is None:
            raise ValueError('mean and scale must be specified')
        print('Data format check passed!!!')

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r  # km

    @staticmethod
    def _div_context_target(train_station_index, num_target):
        """
        For training phase: divide training stations into context and target
        Args:
            train_station_index (ndarray): training node indexes
            num_target: number of target nodes

        Returns:
            target_index (ndarray): target station indexes
            context_index (ndarray): context station indexes
        """
        target_index = np.random.choice(train_station_index, num_target, replace=False)
        context_index = np.setdiff1d(train_station_index, target_index)
        return target_index, context_index

    @staticmethod
    def _get_context_target_index(context_station_list, target_station_list):
        """
        For testing phase: get context and target station index
        Args:
            context_station_list (list): list of context station names
            target_station_list (list): list of target station names

        Returns:
            context_station_index (list): list of context station index
            target_station_index (list): list of target station index
        """
        station_list = context_station_list + target_station_list
        station = np.array(station_list)
        ind = np.argsort(np.argsort(station))
        assert ind.shape[0] == len(station_list)
        context_index = ind[:len(context_station_list)].tolist()
        target_index = ind[len(context_station_list):].tolist()
        return context_index, target_index

    @staticmethod
    def _get_start_index(index, t_len, phase='train'):
        """
        Get the start index of the time series
        Training phase: current index + t_len
        Test phase: no overlap between time series
        Args:
            index (int): index of the time series
            t_len (int): length of the time series
            phase (str): phase of the model

        Returns:
            start_index (int): start index of the time series
            start_index (int): end index of the time series
        """
        if phase == 'train':
            start_index = index
            end_index = index + t_len
        else:
            start_index = index * t_len
            end_index = start_index + t_len
        return start_index, end_index

    @staticmethod
    def _fetch_data_item_from_dict(data, start_index, end_index, station_index=None):
        """
        Fetch data from the time series
        Args:
            data (dict: {'feat'(optional), 'missing', 'pred'}): time series data dictionary.
            Key feat is optional, depending on the dataset.
            station_index (ndarray): station indexes
            start_index (int): start index of the time series
            end_index (int): end index of the time series

        Returns:
            data_item (tensor): data of the time series
        """
        if station_index is None:
            # return all stations
            station_index = np.arange(data['missing'].shape[0])
        pred = torch.from_numpy(data['pred'][station_index, start_index:end_index])
        feat = torch.from_numpy(data['feat'][station_index, start_index:end_index]) if 'feat' in data.keys() else None
        missing = torch.from_numpy(data['missing'][station_index, start_index:end_index])
        # print(data['pred'].shape, data['wind'].shape)
        # print(data['pred'][0].shape, data['wind'][0].shape)
        # print("Done")
        # exit()
        wind = torch.from_numpy(data['wind'][:, start_index:end_index])
        return pred, feat, missing, wind

    @staticmethod
    def _fetch_divided_form_data_item(
            data,
            A,
            index,
            t_len,
            train_node_index,
            test_node_index,
            num_train_target=None,  # training parameter
            phase='train',
    ):
        """
        data will be divided into context and target set, following the setting of neural processes
        Args:
            data: time series data dict (dict{'name': {'feat', 'missing', 'pred'}})
            A: adjacency matrix
            index: int
            t_len: length of time series (int)
            num_train_target: number of target nodes for training (int)
            train_node_index: training node indexes (ndarray)
            test_node_index: testing node indexes (ndarray)
            phase: train, val, test (string)

        Returns:
            data item dictionary (dict {
            'x_context', 'y_context', 'x_target',
            'y_target', 'adj', 'missing_index_context',
            'missing_index_target', 'time'
            })
        """
        if phase == 'train':
            # random divide nodes into context set and target set
            target_index, context_index = BaseDataset._div_context_target(train_node_index, num_train_target)
        else:
            target_index, context_index = test_node_index, train_node_index

        # A_1hop = A[target_index, :][:, context_index][np.newaxis]  # 1-hop neighbor
        # A_2hop = np.dot(A, A)[target_index, :][:, context_index][np.newaxis]  # 2-hop neighbor
        # adj = np.concatenate([A_1hop, A_2hop], axis=0)

        start_index, end_index = BaseDataset._get_start_index(index, t_len, phase)

        pred_target, feat_target, missing_mask_target, _ = BaseDataset._fetch_data_item_from_dict(data, start_index, end_index, target_index)
        pred_context, feat_context, missing_mask_context, wind_data = BaseDataset._fetch_data_item_from_dict(data, start_index, end_index, context_index)

        time = data['time'][start_index:end_index]

        missing_mask_context = missing_mask_context.squeeze(-1)
        missing_mask_target = missing_mask_target.squeeze(-1)
        # adj = torch.from_numpy(adj)

        direction_mapping = {
            0: 0.0,                  # no wind
            1: np.pi / 2,           # east
            2: 3 * np.pi / 2,       # west
            3: np.pi,               # south
            4: 0.0,                  # north
            9: 0.0,                  # unstable (no specific angle)
            13: np.pi / 4,          # southeast
            14: 3 * np.pi / 4,      # northeast
            23: 5 * np.pi / 4,      # southwest
            24: 7 * np.pi / 4       # northwest
        }

        # wind_context has shape (35, 24, 1)
        # wind_context_list = torch.flatten(wind_context)

        # Reshape the tensor and take the mode
        # wind_data = wind_data.reshape(35, 24)  # Reshape to (35, 24)
        # wind_mode, _ = torch.mode(wind_data, dim=1)  # Take mode along dim 1

        # wind_mode = wind_mode.reshape(35, 1)  # Reshape to (35, 1)
        # wind_mode = torch.tensor([direction_mapping[val.item()] for val in wind_mode]).reshape(-1, 1) # replace by directions

        # # Take mean of u and v components
        # wind_directions = [direction_mapping[int(dirn)] for dirn in wind_context_list]
        # u_east = np.mean(np.sin(wind_directions))
        # u_north = np.mean(np.cos(wind_directions))
        # mean_wind_direction = np.arctan2(u_east, u_north)        
        # wind_matrix = np.array([[np.sin(mean_wind_direction)**2,  -np.sin(mean_wind_direction) * np.cos(mean_wind_direction)],
        #                         [-np.sin(mean_wind_direction) * np.cos(mean_wind_direction), np.cos(mean_wind_direction) ** 2]])

        # Take Mode
        wind_context_list = wind_data.flatten()
        wind_mode, counts = torch.mode(wind_context_list)
        mode_wind_direction = direction_mapping[int(wind_mode)]
        wind_matrix = np.array([[np.sin(mode_wind_direction)**2,  -np.sin(mode_wind_direction) * np.cos(mode_wind_direction)],
                                [-np.sin(mode_wind_direction) * np.cos(mode_wind_direction), np.cos(mode_wind_direction) ** 2]])
        
        # wind matrix is "A" in the paper, a 2x2 matrix
        # edited the load_loc function to return A with shape (35, 2) which is latitude and longitude of stations

        # wind_A = np.zeros((A.shape[0], A.shape[0]))
        # for xi in range(A.shape[0]):
        #     for xj in range(A.shape[0]):
        #         tau = np.array([A[xi, 0] - A[xj, 0], A[xi, 1] - A[xj, 1]]).reshape(2, 1)
        #         x = (tau.T @ wind_matrix @ tau)
        #         # print(x.shape)
        #         # print(np.exp(-0.5*x))
        #         wind_A[xi, xj] = np.exp(-0.5*x)

        diff = A[:, np.newaxis, :2] - A[:, :2]
        tau = diff.reshape(A.shape[0], A.shape[0], 2, 1)
        x = np.sum(tau * wind_matrix @ tau, axis=(2, 3))
        wind_A = np.exp(-0.5 * x)

        final_A = wind_A

        batch_data = { 'pred_context': pred_context.float(),  # [num_n, time, d_y]
                 'pred_target': pred_target.float(),  # [num_m, time, d_y]
                #  'adj': adj.float(),  # [2, num_m, num_n]
                 'missing_mask_context': missing_mask_context.float(),  # [num_n, time]
                 'missing_mask_target': missing_mask_target.float(),  # [num_m, time]
                 'time': time,  # [time]
                 'A':torch.from_numpy(final_A),
                 'target_index': torch.tensor(target_index),
                 'context_index': torch.tensor(context_index), 
                 'wind_mode': wind_mode
         }
        # add features if available
        if feat_context is not None:
            batch_data['feat_context'] = feat_context.float() # [num_n, time, d_x]
            batch_data['feat_target'] = feat_target.float() # [num_m, time, d_x]
        return batch_data

    def get_node_division(self, test_nodes_path, num_nodes=None, test_node_ratio=3/10):
        if os.path.isfile(test_nodes_path):
            test_nodes = np.load(test_nodes_path)
            print('Test Nodes are: ', test_nodes)
        else:
            print('No testing nodes. Randomly divide nodes for testing!')
            rand = np.random.RandomState(0)  # Fixed random output
            # test_nodes = rand.choice(list(range(0, num_nodes)), int(num_nodes * test_node_ratio), replace=False)
            test_nodes = rand.choice(list(range(0, num_nodes)), 12, replace = False)
            print('Test Nodes are: ', test_nodes)
            np.save(test_nodes_path, test_nodes)
        return test_nodes
