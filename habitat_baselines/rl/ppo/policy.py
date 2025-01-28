#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
import numpy as np
from numpy.lib import ufunclike
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.common.utils import CategoricalNet, Flatten, to_grid
from habitat_baselines.rl.models.projection import Projection, RotateTensor, get_grid
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import RGBCNNNonOracle, RGBCNNOracle, MapCNN, SEMSEGCNNNonOracle
from habitat_baselines.rl.models.semantic_map_utils import get_network_from_options, run_img_segm
from habitat_baselines.rl.models.goal_map_utils import GoalDetector
from typing import Optional
from torch import Tensor

from habitat_baselines.rl.ppo.aux_losses_utils import (
    get_obj_poses,
    compute_distance_labels,
    compute_direction_labels,
)


class PolicyNonOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        deterministic=False,
        nb_steps=None,
        current_episodes=None,
    ):
        features, goal_image, rnn_hidden_states, global_map = self.net(
            observations,
            None,
            rnn_hidden_states,
            global_map,
            prev_actions,
            masks,
            nb_steps,
            current_episodes,
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, global_map, goal_image

    def get_value(
        self, observations, goal_image, rnn_hidden_states, global_map, prev_actions, masks
    ):
        features, _, _, _ = self.net(
            observations, goal_image, rnn_hidden_states, global_map, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, goal_image, rnn_hidden_states, global_map, prev_actions, masks, action
    ):
        (
            features,
            rnn_hidden_states,
            global_map,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        ) = self.net(
            observations,
            # segmentation,
            goal_image,
            rnn_hidden_states,
            global_map,
            prev_actions,
            masks,
            ev=1,
            aux_loss=True,
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        )


class PolicyOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, *_ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, *_ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        (
            features,
            rnn_hidden_states,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        ) = self.net(observations, rnn_hidden_states, prev_actions, masks, True)

        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        )


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class BaselinePolicyNonOracle(PolicyNonOracle):
    def __init__(
        self,
        batch_size,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        global_map_semantic,
        global_map_goal,
        configs,
        coordinate_min,
        coordinate_max,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetNonOracle(
                batch_size,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                egocentric_map_size=egocentric_map_size,
                global_map_size=global_map_size,
                global_map_depth=global_map_depth,
                global_map_semantic=global_map_semantic,
                global_map_goal=global_map_goal,
                configs=configs,
                coordinate_min=coordinate_min,
                coordinate_max=coordinate_max,
                aux_loss_seen_coef=aux_loss_seen_coef,
                aux_loss_direction_coef=aux_loss_direction_coef,
                aux_loss_distance_coef=aux_loss_distance_coef,
            ),
            action_space.n,
        )


class BaselinePolicyOracle(PolicyOracle):
    def __init__(
        self,
        agent_type,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetOracle(
                agent_type,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                aux_loss_seen_coef=aux_loss_seen_coef,
                aux_loss_direction_coef=aux_loss_direction_coef,
                aux_loss_distance_coef=aux_loss_distance_coef,
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, global_map, prev_actions):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class BaselineNetNonOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        batch_size,
        observation_space,
        hidden_size,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        global_map_semantic,
        global_map_goal,
        configs,
        coordinate_min,
        coordinate_max,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth
        self.global_map_semantic = global_map_semantic
        self.global_map_goal = global_map_goal
        self.configs = configs

        self.visual_encoder = RGBCNNNonOracle(observation_space, hidden_size)

        # self.img_segmentor = get_network_from_options(self.configs)
        # self.semantic_encoder = SEMSEGCNNNonOracle(observation_space, self.device, hidden_size)

        self.goal_detector = GoalDetector(self.device)
        self.resizing = nn.Upsample(size=(12, 12), mode='bilinear', align_corners=False)
        # self.resizing_sem = nn.Upsample(size=(12, 12), mode='bilinear', align_corners=False)
        # self.resizing = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)
        # self.resizing_sem = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

        # need to revise
        # for 128x128
        # self.goal_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=False),
        #     nn.MaxPool2d(kernel_size=2, stride=1),
        #     nn.ReLU(inplace=True),
        # )
        # for 256x256
        # self.goal_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, bias=False),
        #     # nn.MaxPool2d(kernel_size=2, stride=1),
        #     nn.ReLU(inplace=True),
        # )

        self.map_encoder = MapCNN(51, 256, "non-oracle")

        self.point_wise = nn.Conv2d(32+8, 16, 1, 1) # 32 + 17 + 8
        self.dropout = nn.Dropout(p=0.25)

        self.projection = Projection(
            egocentric_map_size, global_map_size, device, coordinate_min, coordinate_max
        )

        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.rotate_tensor = RotateTensor(device)

        # self.image_features_linear = nn.Linear(32 * 28 * 28, 512)
        self.image_features_linear = nn.Linear(16 * 12 * 12, 512)  # 12, 12 rgb-d + semantic
        # self.global_features_linear = nn.Linear(16 * 7 * 7, 256)  # rgb-d + semantic

        self.flatten = Flatten()

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                self._hidden_size
                + 256
                + object_category_embedding_size
                + previous_action_embedding_size,
                self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (0 if self.is_blind else self._hidden_size)
                + object_category_embedding_size,
                self._hidden_size,  # Replace 2 by number of target categories later
            )
        self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        self.full_global_map = torch.zeros(
            batch_size,
            global_map_size,
            global_map_size,
            global_map_depth + global_map_semantic + global_map_goal,
            device=self.device,
        )

        self.layer_init()

        # Auxiliary losses
        if aux_loss_seen_coef is not None:
            self.fc_seen = nn.Linear(self._hidden_size, 2)
            nn.init.orthogonal_(self.fc_seen.weight)
            nn.init.constant_(self.fc_seen.bias, 0)
        else:
            self.fc_seen = None

        if aux_loss_direction_coef is not None:
            self.fc_direction = nn.Linear(self._hidden_size, 12)
            nn.init.orthogonal_(self.fc_direction.weight)
            nn.init.constant_(self.fc_direction.bias, 0)
        else:
            self.fc_direction = None

        if aux_loss_distance_coef is not None:
            self.fc_distance = nn.Linear(self._hidden_size, 35)
            nn.init.orthogonal_(self.fc_distance.weight)
            nn.init.constant_(self.fc_distance.bias, 0)
        else:
            self.fc_distance = None

        self.train()

    def layer_init(self):
        # for name, param in self.goal_encoder.named_parameters():
        #     if "weight" in name:
        #         nn.init.orthogonal_(param)
        #     elif "bias" in name:
        #         nn.init.constant_(param, 0)
        for name, param in self.point_wise.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        nn.init.orthogonal_(self.goal_embedding.weight)
        nn.init.orthogonal_(self.action_embedding.weight)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(
        self,
        observations,
        # semantic_segmentation,
        masked_goal_image,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        nb_steps=None,
        current_episodes=None,
        ev=0,
        aux_loss=False,
    ):
        target_encoding = self.get_target_encoding(observations)
        goal_embed = self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)

        loss_seen = None
        pred_softmax = None
        loss_directions = None
        loss_distances = None
        seen_labels = None

        if aux_loss and (
            (self.fc_seen is not None)
            or (self.fc_direction is not None)
            or (self.fc_distance is not None)
        ):
            # Get positions of target objs
            mean_i_obj, mean_j_obj, gt_seen, not_visible_goals = get_obj_poses(
                observations
            )

            # Compute euclidian distance
            if self.fc_distance is not None:
                distance_labels = compute_distance_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

            if self.fc_seen is not None:
                seen_labels = np.array(gt_seen).astype(np.int_)
                seen_labels = torch.from_numpy(seen_labels).to(self.device)

            # Compute directions
            if self.fc_direction is not None:
                direction_labels = compute_direction_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)

        grid_x, grid_y = self.to_grid.get_grid_coords(observations["gps"])
        bs = global_map.shape[0]
        ##forward pass specific
        if ev == 0:
            # pred_ssegs = (run_img_segm(model=self.img_segmentor, input_obs=observations) + 9) / 26.0  # [16, 256, 256, 27]
            masked_goal_image = self.goal_detector.localize_goal(observations)  # / 26.0  # 8.0  # [bs, 8, 28, 28]
            # obstacle_embed = self.resizing_sem(pred_ssegs)
            # obstacle_embed = self.semantic_encoder(pred_ssegs)
            goal_image_embed = self.resizing(masked_goal_image)

            for i in range(goal_image_embed.shape[1]):  # make label
                goal_image_embed[:, i] = goal_image_embed[:, i] * (i + 1)
            goal_image_embed = goal_image_embed / 26.0  # normalize semantic goal label

            perception_embed_projection = torch.cat([perception_embed, goal_image_embed], dim=1)  # 16, 64, 28, 28 # obstacle_embed,
            projection = self.projection.forward(
                perception_embed_projection, observations["depth"] * 10, -(observations["compass"])
            )

            ''' Feature Fusion Module '''
            # (1) channel-wise concatenate
            mixed_embed_local = torch.cat([perception_embed, goal_image_embed], dim=1)  # 16, 64, 12, 12 # obstacle_embed,
            # mixed_embed_local = torch.cat([perception_embed, obstacle_embed], dim=1)
            # (2) point-wise convolution (fusion)
            mixed_embed_local = F.relu(self.point_wise(mixed_embed_local))
            mixed_embed_local = self.dropout(mixed_embed_local)
            # (3) flatten & FC layer
            perception_embed = self.image_features_linear(self.flatten(mixed_embed_local))
            ''' END '''

            self.full_global_map[:bs, :, :, :] = self.full_global_map[
                :bs, :, :, :
            ] * masks.unsqueeze(1).unsqueeze(1)
            if bs != 12:
                self.full_global_map[bs:, :, :, :] = (
                    self.full_global_map[bs:, :, :, :] * 0
                )
            if torch.cuda.is_available():
                with torch.cuda.device(1):
                    agent_view = torch.cuda.FloatTensor(
                        bs,
                        self.global_map_depth + self.global_map_semantic + self.global_map_goal,
                        self.global_map_size,
                        self.global_map_size,
                    ).fill_(0)
            else:
                agent_view = (
                    torch.FloatTensor(
                        bs,
                        self.global_map_depth + self.global_map_semantic + self.global_map_goal,
                        self.global_map_size,
                        self.global_map_size,
                    )
                    .to(self.device)
                    .fill_(0)
                )
            agent_view[
                :,
                :,
                self.global_map_size // 2
                - math.floor(self.egocentric_map_size / 2) : self.global_map_size // 2
                + math.ceil(self.egocentric_map_size / 2),
                self.global_map_size // 2
                - math.floor(self.egocentric_map_size / 2) : self.global_map_size // 2
                + math.ceil(self.egocentric_map_size / 2),
            ] = projection
            st_pose = torch.cat(
                [
                    -(grid_y.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    -(grid_x.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    observations["compass"],
                ],
                dim=1,
            )

            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
            rotated = F.grid_sample(agent_view, rot_mat)
            translated = F.grid_sample(rotated, trans_mat)

            self.full_global_map[:bs, :, :, :] = torch.max(
                self.full_global_map[:bs, :, :, :], translated.permute(0, 2, 3, 1)
            )
            st_pose_retrieval = torch.cat(
                [
                    (grid_y.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    (grid_x.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    torch.zeros_like(observations["compass"]),
                ],
                dim=1,
            )
            _, trans_mat_retrieval = get_grid(
                st_pose_retrieval, agent_view.size(), self.device
            )
            translated_retrieval = F.grid_sample(
                self.full_global_map[:bs, :, :, :].permute(0, 3, 1, 2),
                trans_mat_retrieval,
            )
            translated_retrieval = translated_retrieval[
                :,
                :,
                self.global_map_size // 2
                - math.floor(51 / 2) : self.global_map_size // 2
                + math.ceil(51 / 2),
                self.global_map_size // 2
                - math.floor(51 / 2) : self.global_map_size // 2
                + math.ceil(51 / 2),
            ]
            final_retrieval = self.rotate_tensor.forward(
                translated_retrieval, observations["compass"]
            )

            global_map_embed = self.map_encoder(final_retrieval.permute(0, 2, 3, 1))

            # ''' Feature Fusion Module '''
            # # (2) point-wise convolution (fusion)
            # mixed_embed_global = F.relu(self.point_wise(global_map_embed))
            # mixed_embed_global = self.dropout(mixed_embed_global)
            # # (3) flatten & FC layer
            # global_map_embed = self.global_features_linear(self.flatten(mixed_embed_global))
            # ''' END '''

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat(
                (perception_embed, global_map_embed, goal_embed, action_embedding),
                dim=1,
            )
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

            return x, masked_goal_image.permute(0, 2, 3, 1), rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1) # pred_ssegs.permute(0, 2, 3, 1),
        else:
            # semantic_segmentation
            from habitat_baselines import test_hyeon_utils
            import matplotlib.pyplot as plt

            # pred_ssegs = semantic_segmentation.permute(0, 3, 1, 2) # B x W x H x C -> B x C x W x H : 0, 3, 1, 2
            masked_goal_image = masked_goal_image.permute(0, 3, 1, 2)

            # masked_goal_image = masked_goal_image  # / 26.0  # 8.0  # [bs, 8, 28, 28]
            #
            # for visualization
            # print("pred_ssegs", pred_ssegs.shape)
            # img_labels = torch.argmax(pred_ssegs, dim=1, keepdim=True)  # B x 1 x cH x cW .detach()
            # goal_labels = torch.argmax(masked_goal_image, dim=1, keepdim=True)  # B x 1 x cH x cW .detach()
            # vis = test_hyeon_utils.colorize_grid(img_labels).squeeze()
            # goal = test_hyeon_utils.colorize_grid(goal_labels).squeeze()
            # for i in range(pred_ssegs.shape[0]):  # batch
            #     plt.imsave('visualization/rgb/rgb{0}.png'.format(i), np.array(observations['rgb'][i].squeeze().cpu(), dtype=np.uint8))
            #     plt.imsave('visualization/goal/goal{0}.png'.format(i), np.array(goal[i].squeeze().cpu(), dtype=np.uint8))
            #     # plt.imsave('visualization/semantic/sem{0}.png'.format(i), np.array(img_labels.squeeze().cpu()))
            #     plt.imsave('visualization/semantic/sem{0}.png'.format(i), np.array(vis[i].squeeze().cpu()))
            #     plt.imsave('visualization/semGT/semGT{0}.png'.format(i), np.array(observations['semantic'][i].squeeze().cpu(), dtype=np.uint8))
            # import pdb
            # pdb.set_trace()
            # obstacle_embed = self.resizing_sem(pred_ssegs)
            # obstacle_embed = self.semantic_encoder(pred_ssegs)
            goal_image_embed = self.resizing(masked_goal_image)

            for i in range(goal_image_embed.shape[1]):  # make label
                goal_image_embed[:, i] = goal_image_embed[:, i] * (i + 1)
            goal_image_embed = goal_image_embed / 26.0  # normalize semantic goal label

            perception_embed_projection = torch.cat([perception_embed, goal_image_embed], dim=1)  # 16, 64, 28, 28 # obstacle_embed,
            projection = self.projection.forward(
                perception_embed_projection, observations["depth"] * 10, -(observations["compass"])
            )

            ''' Feature Fusion Module '''
            # (1) channel-wise concatenate
            mixed_embed_local = torch.cat([perception_embed, goal_image_embed], dim=1)  # 16, 64, 12, 12 #  obstacle_embed,
            # mixed_embed_local = torch.cat([perception_embed, obstacle_embed], dim=1)
            # (2) point-wise convolution (fusion)
            mixed_embed_local = F.relu(self.point_wise(mixed_embed_local))
            mixed_embed_local = self.dropout(mixed_embed_local)
            # (3) flatten & FC layer
            perception_embed = self.image_features_linear(self.flatten(mixed_embed_local))
            ''' END '''

            global_map = global_map * masks.unsqueeze(1).unsqueeze(1)
            with torch.cuda.device(1):
                agent_view = torch.cuda.FloatTensor(
                    bs, self.global_map_depth + self.global_map_semantic + self.global_map_goal, 51, 51
                ).fill_(0)
            agent_view[
                :,
                :,
                51 // 2
                - math.floor(self.egocentric_map_size / 2) : 51 // 2
                + math.ceil(self.egocentric_map_size / 2),
                51 // 2
                - math.floor(self.egocentric_map_size / 2) : 51 // 2
                + math.ceil(self.egocentric_map_size / 2),
            ] = projection

            final_retrieval = torch.max(global_map, agent_view.permute(0, 2, 3, 1))

            global_map_embed = self.map_encoder(final_retrieval)

            # ''' Feature Fusion Module '''
            # # (2) point-wise convolution (fusion)
            # mixed_embed_global = F.relu(self.point_wise(global_map_embed))
            # mixed_embed_global = self.dropout(mixed_embed_global)
            # # (3) flatten & FC layer
            # global_map_embed = self.global_features_linear(self.flatten(mixed_embed_global))
            # ''' END '''

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat(
                (perception_embed, global_map_embed, goal_embed, action_embedding),
                dim=1,
            )
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

            # Compute auxiliary losses
            if aux_loss and (
                (self.fc_seen is not None)
                or (self.fc_direction is not None)
                or (self.fc_distance is not None)
            ):
                if self.fc_seen is not None:
                    pred_seen = self.fc_seen(x)
                    loss_seen = F.cross_entropy(pred_seen, seen_labels)
                    pred_softmax = F.softmax(pred_seen, dim=1)

                indices_to_keep = (seen_labels == 1).nonzero()[:, 0]
                if len(indices_to_keep) > 0:
                    if self.fc_direction is not None:
                        pred_directions = self.fc_direction(x)
                        loss_directions = F.cross_entropy(
                            pred_directions[indices_to_keep],
                            direction_labels[indices_to_keep],
                        )

                    if self.fc_distance is not None:
                        pred_distances = self.fc_distance(x)
                        loss_distances = F.cross_entropy(
                            pred_distances[indices_to_keep],
                            distance_labels[indices_to_keep],
                        )
                else:
                    if self.fc_direction is not None:
                        loss_directions = torch.zeros(1).squeeze(0).to(self.device)
                    if self.fc_distance is not None:
                        loss_distances = torch.zeros(1).squeeze(0).to(self.device)

            return (
                x,
                rnn_hidden_states,
                final_retrieval.permute(0, 2, 3, 1),
                loss_seen,
                loss_directions,
                loss_distances,
                pred_softmax,
                seen_labels,
            )


class BaselineNetOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        agent_type,
        observation_space,
        hidden_size,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
    ):
        super().__init__()
        self.agent_type = agent_type
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action

        self.visual_encoder = RGBCNNOracle(observation_space, 512)
        if agent_type == "oracle":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.occupancy_embedding = nn.Embedding(3, 16)
            self.object_embedding = nn.Embedding(9, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)

        elif agent_type == "no-map":
            self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        elif agent_type == "oracle-ego":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.object_embedding = nn.Embedding(10, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)

        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size)
                + object_category_embedding_size
                + previous_action_embedding_size,
                self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size) + object_category_embedding_size,
                self._hidden_size,
            )

        # Auxiliary losses
        if aux_loss_seen_coef is not None:
            self.fc_seen = nn.Linear(self._hidden_size, 2)
            nn.init.orthogonal_(self.fc_seen.weight)
            nn.init.constant_(self.fc_seen.bias, 0)
        else:
            self.fc_seen = None

        if aux_loss_direction_coef is not None:
            self.fc_direction = nn.Linear(self._hidden_size, 12)
            nn.init.orthogonal_(self.fc_direction.weight)
            nn.init.constant_(self.fc_direction.bias, 0)
        else:
            self.fc_direction = None

        if aux_loss_distance_coef is not None:
            self.fc_distance = nn.Linear(self._hidden_size, 35)
            nn.init.orthogonal_(self.fc_distance.weight)
            nn.init.constant_(self.fc_distance.bias, 0)
        else:
            self.fc_distance = None

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(
        self, observations, rnn_hidden_states, prev_actions, masks, aux_loss=False
    ):
        target_encoding = self.get_target_encoding(observations)
        x = [
            self.goal_embedding(
                (target_encoding).type(torch.LongTensor).to(self.device)
            ).squeeze(1)
        ]
        bs = target_encoding.shape[0]
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        loss_seen = None
        pred_softmax = None
        loss_directions = None
        loss_distances = None
        seen_labels = None

        if aux_loss and (
            (self.fc_seen is not None)
            or (self.fc_direction is not None)
            or (self.fc_distance is not None)
        ):
            # Get positions of target objs
            mean_i_obj, mean_j_obj, gt_seen, not_visible_goals = get_obj_poses(
                observations
            )

            # Compute euclidian distance
            if self.fc_distance is not None:
                distance_labels = compute_distance_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

            if self.fc_seen is not None:
                seen_labels = np.array(gt_seen).astype(np.int_)
                seen_labels = torch.from_numpy(seen_labels).to(self.device)

            # Compute directions
            if self.fc_direction is not None:
                direction_labels = compute_direction_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

        if self.agent_type != "no-map":
            global_map_embedding = []
            global_map = observations["semMap"]
            if self.agent_type == "oracle":
                global_map_embedding.append(
                    self.occupancy_embedding(
                        global_map[:, :, :, 0]
                        .type(torch.LongTensor)
                        .to(self.device)
                        .view(-1)
                    ).view(bs, 50, 50, -1)
                )

            global_map_embedding.append(
                self.object_embedding(
                    global_map[:, :, :, 1]
                    .type(torch.LongTensor)
                    .to(self.device)
                    .view(-1)
                ).view(bs, 50, 50, -1)
            )
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x

        if self.use_previous_action:
            x = torch.cat(x + [self.action_embedding(prev_actions).squeeze(1)], dim=1)
        else:
            x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        # Compute auxiliary losses
        if aux_loss and (
            (self.fc_seen is not None)
            or (self.fc_direction is not None)
            or (self.fc_distance is not None)
        ):
            if self.fc_seen is not None:
                pred_seen = self.fc_seen(x)
                loss_seen = F.cross_entropy(pred_seen, seen_labels)
                pred_softmax = F.softmax(pred_seen, dim=1)

            indices_to_keep = (seen_labels == 1).nonzero()[:, 0]
            if len(indices_to_keep) > 0:
                if self.fc_direction is not None:
                    pred_directions = self.fc_direction(x)
                    loss_directions = F.cross_entropy(
                        pred_directions[indices_to_keep],
                        direction_labels[indices_to_keep],
                    )

                if self.fc_distance is not None:
                    pred_distances = self.fc_distance(x)
                    loss_distances = F.cross_entropy(
                        pred_distances[indices_to_keep],
                        distance_labels[indices_to_keep],
                    )
            else:
                if self.fc_direction is not None:
                    loss_directions = torch.zeros(1).squeeze(0).to(self.device)
                if self.fc_distance is not None:
                    loss_distances = torch.zeros(1).squeeze(0).to(self.device)

        return (
            x,
            rnn_hidden_states,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_softmax,
            seen_labels,
        )
