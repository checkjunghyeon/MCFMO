import torch
import torch.nn as nn
import skimage
import skimage.measure


class GoalDetector():
    def __init__(self, device):
        self.device = device
        self.goal_dict = self.fill_goal_dict()
        self.goal_features = self.fill_features()
        self.goal_measure = 'threshold'
        # self.resizing = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)

    def reset(self):
        self.goal_map.fill_(0.0)
        self.agent_pos = (self.map_size // 2, self.map_size // 2)  # H, W position

    def localize_goal(self, observations):   #in local view
        rgb = (observations["rgb"].float() / 255.)  # (B, H, W, C)
        depth = observations["depth"].clone().detach()

        # rgb = self.resizing(rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # depth = self.resizing(depth.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # compass = observations["compass"].clone().detach().fill_(0.0)  # This should not apply
        goal_image = torch.zeros(rgb.shape[0], rgb.shape[1], rgb.shape[2], 8, device=self.device)  # (B, H, W, 8)

        for idx, goal_feature in enumerate(self.goal_features):
            diff = torch.norm(rgb - goal_feature, dim=-1)  # (B, H, W)
            if idx == 4 or idx == 6:  # white, black
                threshold = 0.0001
            elif idx == 2 or idx == 1 or idx == 0 or idx == 3:  # blue, green, red, yellow
                threshold = 0.1
            else:  # cyan, pink, red
                threshold = 0.5

            # Quantize diff with threshold
            inlier = (diff < threshold) & (depth != 0).squeeze(-1)
            outlier = (diff > threshold) | (depth == 0).squeeze(-1)
            diff[inlier] = 1.
            diff[outlier] = 0.

            size_threshold_minimum = 40  # 10
            size_threshold_maximum = 800 # tmp
            labels, num = skimage.measure.label(diff.squeeze(0).cpu().int().numpy(), connectivity=2, return_num=True)

            # for label_idx in range(1, num + 1):  # Non-zero labels
            #     if (labels == label_idx).sum() > size_threshold:
            #         inlier = torch.from_numpy((labels == label_idx)) #.unsqueeze(0)  # (B, H, W)
            #         if inlier.size(0) == 28:
            #             inlier = inlier.unsqueeze(0)
            #         diff[inlier] = 1.0
            #         diff[~inlier] = 0.0
            #         break
            #     else:
            #         diff.fill_(0.0)

            for label_idx in range(1, num + 1):  # Non-zero labels
                if size_threshold_minimum < (labels == label_idx).sum() < size_threshold_maximum:
                    # print("- num:", (labels == label_idx).sum())
                    inlier = torch.from_numpy((labels == label_idx)) # .unsqueeze(0)  # (B, H, W)
                    diff[inlier] = 1.0
                    diff[~inlier] = 0.0
                    break
                else:
                    diff.fill_(0.0)

            goal_image[..., idx] = diff

        goal_image = goal_image.permute(0, 3, 1, 2)

        return goal_image

    def check_current_goal(self, observations):
        """
        Decide if current goal is localized in self.goal_map.

        Args:
            observations: Dictionary containing observations.

        Returns:
            goal_localized: True if goal is localized.
        """
        tgt_goal_idx = observations['multiobjectgoal'].item()
        print("observations['multiobjectgoal']", observations['multiobjectgoal'])
        if self.goal_measure == 'threshold':
            # Check if any grid in self.goal_map is over threshold
            goal_localized = torch.any(self.goal_map[..., tgt_goal_idx] > self.goal_threshold).bool().item()
            return goal_localized

        else:
            raise ValueError("Invalid goal measure")

    def find_current_goal(self, observations=None, tgt_goal_idx=None):
        """
        Find the location of current goal in self.goal_map.

        Args:
            observations: Dictionary containing observations.
            tgt_goal_idx: Index of goal to look for.

        Returns:
            goal_position: torch.tensor of shape (2, ) containing goal position
        """
        assert observations is not None or tgt_goal_idx is not None

        if tgt_goal_idx is None:
            tgt_goal_idx = observations['multiobjectgoal'].item()

        if self.goal_measure == 'threshold':
            # Average grid locations in self.goal_map that are over threshold
            over_thres = torch.where(
                (self.goal_map[..., tgt_goal_idx] > self.goal_threshold).squeeze(0))  # Indices for H, W

            goal_position = torch.tensor([over_thres[0].float().mean(), over_thres[1].float().mean()],
                                         device=self.device).long()  # H, W position
            return goal_position
        else:
            raise ValueError("Invalid goal measure")

    def fill_features(self):
        """
        Fill self.goal_features with goal-related prior information.

        Args:
            None

        Returns:
            goal_features: (N_goal, N_features) tensor containing goal feature information
        """
        return torch.tensor([
            [1., 0., 0.0017],  # red
            [0., 0.1897, 0.], # green
            [0.0018, 0.0037, 0.5288],  # blue
            [1., 1., 1.],  # white
            [0.969, 0.0001, 1.],  # pink
            [0., 0., 0.],  # black
            [0., 1., 1.]  # cyan
        ], device=self.device)

    def fill_goal_dict(self):
        """
        Fill self.goal_dict with goal-related prior information.

        Args:
            None

        Returns:
            goal_dict: Dictionary containing mapping from goal name to goal index
        """

        return {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2,
            'cylinder_yellow':3, 'cylinder_white':4, 'cylinder_pink':5, 'cylinder_black':6, 'cylinder_cyan':7
        }