import torch
import yaml
import xml.etree.ElementTree as ET

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

class TendonRobotModel:
    def __init__(self, yaml_path, urdf_path, num_envs, device):
        """
        Args:
            yaml_path (str): パラメータファイル（params.yaml）のパス
            urdf_path (str): URDFファイルのパス
            num_envs (int): 環境数（バッチサイズ）
            device (torch.device): GPUなどのデバイス
        """
        self.device = device
        self.num_envs = num_envs

        # YAMLからジョイント情報とワイヤ情報を読み込む
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        self.joint_names = params["JointList"]
        self.num_joints = len(self.joint_names)
        
        tendon_list = params["TendonList"]
        self.num_tendons = len(tendon_list)
        # 各ワイヤについて、viaの名前リストを保持
        self.tendon_via_names = []
        self.affected_joints = []
        for tendon_info in tendon_list:
            via_points = tendon_info["ViaPoints"]
            via_names = [via_info["ParentLink"] for via_info in via_points]
            self.tendon_via_names.append(via_names)
            self.affected_joints.append(tendon_info["AffectedJoints"])
        # バッチ用のジョイントパラメータを初期化
        # 各ジョイントは shape=(num_envs, 3) のテンソルで管理
        self.joint_origin_init = torch.zeros((num_envs, self.num_joints, 3), dtype=torch.float32, device=device)
        self.joint_axis_init = torch.zeros((num_envs, self.num_joints, 3), dtype=torch.float32, device=device)
        # 現在のジョイント状態
        self.joint_origin = torch.zeros((num_envs, self.num_joints, 3), dtype=torch.float32, device=device)
        self.joint_axis = torch.zeros((num_envs, self.num_joints, 3), dtype=torch.float32, device=device)
        # 関節角度／変位
        self.joint_dof_pos = torch.zeros((num_envs, self.num_joints), dtype=torch.float32, device=device)
        # 各ジョイントの種類（"revolute" か "prismatic"/"slide"）
        self.joint_types = ["revolute"] * self.num_joints  # 仮の初期値
        
        # 各ジョイントの親リンク名、子リンク名、および親ジョイントのインデックスを保持するリスト
        self.joint_parent_links = [None] * self.num_joints
        self.joint_child_links  = [None] * self.num_joints
        self.joint_parent_idx   = [-1] * self.num_joints  # -1ならベースリンク（親が存在しない）

        # URDFから各ジョイントのorigin, axis, typeを取得し上書きする
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint_elem in root.findall("joint"):
            jname = joint_elem.get("name")
            if jname in self.joint_names:
                idx = self.joint_names.index(jname)
                jtype = joint_elem.get("type")
                mapped_type = "revolute" if jtype == "revolute" else ("prismatic" if jtype == "prismatic" else jtype)
                self.joint_types[idx] = mapped_type
                # origin情報
                origin_elem = joint_elem.find("origin")
                if origin_elem is not None:
                    origin_str = origin_elem.get("xyz", "0 0 0")
                    origin_list = [float(x) for x in origin_str.split()]
                else:
                    origin_list = [0, 0, 0]
                # axis情報
                axis_elem = joint_elem.find("axis")
                if axis_elem is not None:
                    axis_str = axis_elem.get("xyz", "0 0 1")
                    axis_list = [float(x) for x in axis_str.split()]
                else:
                    axis_list = [0, 0, 1]
                # parent, child情報
                parent_elem = joint_elem.find("parent")
                parent_link = parent_elem.get("link") if parent_elem is not None else None
                child_elem  = joint_elem.find("child")
                child_link  = child_elem.get("link") if child_elem is not None else None
                self.joint_parent_links[idx] = parent_link
                self.joint_child_links[idx]  = child_link
                
                # 各環境に同じ値を設定（タイル展開）
                origin_tensor = torch.tensor(origin_list, dtype=torch.float32, device=device)
                axis_tensor   = torch.tensor(axis_list, dtype=torch.float32, device=device)
                axis_tensor = axis_tensor / torch.norm(axis_tensor, dim=-1, keepdim=True) # 正規化
                self.joint_origin_init[:, idx, :] = origin_tensor.unsqueeze(0).expand(num_envs, -1)
                self.joint_axis_init[:, idx, :] = axis_tensor.unsqueeze(0).expand(num_envs, -1)
                # 初期状態の値としてコピー
                self.joint_origin[:, idx, :] = self.joint_origin_init[:, idx, :]
                self.joint_axis[:, idx, :] = self.joint_axis_init[:, idx, :]

         # 各ジョイントの親子関係の構築（parent_linkがどのジョイントのchild_linkに対応するかを調べる）
        for i in range(self.num_joints):
            parent_link = self.joint_parent_links[i]
            if parent_link is None:
                self.joint_parent_idx[i] = -1
            else:
                found = False
                for j in range(self.num_joints):
                    if self.joint_child_links[j] == parent_link:
                        self.joint_parent_idx[i] = j
                        found = True
                        break
                if not found:
                    self.joint_parent_idx[i] = -1  # 親がベースリンクの場合

        # 各関節の【局所】同時変換行列（親リンクと子リンク間の変換）を保持するテンソル (num_envs, num_joints, 4, 4)
        self.t_joint = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, self.num_joints, 4, 4).clone()
        # 各関節の【グローバル】同時変換行列（ベースリンクと子リンク間の変換）を保持するテンソル (num_envs, num_joints, 4, 4)
        self.t_global = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, self.num_joints, 4, 4).clone()

        # バッチ用のワイヤ（via）の初期化
        # tendon_via_pos はリスト（長さ：num_tendons）、各要素は shape=(num_envs, num_vias, 3) のテンソル. ベースリンク座標系
        self.tendon_via_pos = []
        # tendon_via_indices は後から gym など外部から設定するためのリスト（各viaに対応する剛体のインデックス）
        self.tendon_via_indices = []
        for via_names in self.tendon_via_names:
            num_vias = len(via_names)
            self.tendon_via_pos.append(torch.zeros((num_envs, num_vias, 3), dtype=torch.float32, device=device))
            self.tendon_via_indices.append([None] * num_vias)

        self.tendon_lengths = torch.zeros((num_envs, self.num_tendons), dtype=torch.float32, device=device)
        self.tendon_jacobian = torch.zeros((num_envs, self.num_joints, self.num_tendons), dtype=torch.float32, device=device)

    def update_state(self, dof_pos_tensor, rigid_body_state, root_state):
        """
        各環境分の関節角度・変位および剛体状態から、ジョイントとワイヤ内のviaの位置を一括更新する
        Args:
            dof_pos_tensor: (num_envs, num_joints) の tensor。各環境の各ジョイントの値
            rigid_body_state: (num_envs, num_rigid_bodies, 3) の tensor。各環境の剛体位置
            root_state: (num_envs, 3) の tensor。各環境のベースリンクの状態. [0:3] は位置, [3:7] はクォータニオン, [7:10] は並進速度, [10:13] は角速度
        """
        #jointのaxisとoriginを更新
        self.joint_dof_pos = dof_pos_tensor
        self.calc_transration_matrix()
        self.joint_origin = self.t_global[:, :, :3, 3]
        for i in range(self.num_joints):
            R_global = self.t_global[:, i, :3, :3]
            axis_init = self.joint_axis_init[:, i, :].unsqueeze(-1)
            axis_updated = torch.bmm(R_global, axis_init) 
            self.joint_axis[:, i, :] = axis_updated.squeeze(-1)

        # ワイヤ内の各viaについて、外部から設定済みの rigid_body_state のインデックスを使って位置を更新
        for t, via_names in enumerate(self.tendon_via_names):
            num_vias = len(via_names)
            for i in range(num_vias):
                idx = self.tendon_via_indices[t][i]
                via_pos_global = rigid_body_state[:, idx, 0:3]
                via_pos_base = quat_rotate_inverse(root_state[:, 3:7], via_pos_global - root_state[:, 0:3]) # ベースリンク座標系への変換
                self.tendon_via_pos[t][:, i, :] = via_pos_base
                
        self.calc_tendon_len()
        self.calc_tendon_jacobian()

    def create_translation_matrix(self, t):
        """
        バッチ対応の平行移動行列を作成
        Args:
            t: (num_envs, 3) tensor – 平行移動ベクトル
        Returns:
            T: (num_envs, 4, 4) ホモジニアス変換行列
        """
        T = torch.eye(4, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        T[:, :3, 3] = t
        return T

    def embed_rotation(self, R):
        """
        3x3の回転行列をホモジニアス変換行列に埋め込む
        Args:
            R: (num_envs, 3, 3) tensor – 回転行列
        Returns:
            T: (num_envs, 4, 4) ホモジニアス変換行列
        """
        T = torch.eye(4, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        T[:, :3, :3] = R
        return T

    def rotation_matrix(self, axis, theta):
        """
        バッチ対応の Rodrigues 回転公式
        Args:
            axis: (num_envs, 1, 3) tensor 
            theta: (num_envs, 1) tensor
        Returns:
            R: (num_envs, 3, 3) 回転行列
        """
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-6)
        a_x = axis[..., 0].unsqueeze(-1).unsqueeze(-1)
        a_y = axis[..., 1].unsqueeze(-1).unsqueeze(-1)
        a_z = axis[..., 2].unsqueeze(-1).unsqueeze(-1)
        zero = torch.zeros_like(a_x)
        K = torch.cat([
            torch.cat([zero, -a_z, a_y], dim=-1),
            torch.cat([a_z, zero, -a_x], dim=-1),
            torch.cat([-a_y, a_x, zero], dim=-1)
        ], dim=-2)
        I = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0)
        theta = theta.unsqueeze(-1).unsqueeze(-1)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        return R.squeeze(1)

    def calc_transration_matrix(self):
        """
        各関節の【局所】, 【グローバル】同時変換行列を計算する
        """
        for i in range(self.num_joints):
            # ① URDFで定義されたオフセットからの平行移動行列 T_origin
            t_origin = self.joint_origin_init[:, i, :]  # (num_envs, 3)
            T_origin = self.create_translation_matrix(t_origin)  # (num_envs, 4, 4)

            # ② 関節運動による変換 T_motion
            if self.joint_types[i] == "revolute":
                # 回転の場合：joint_axis_init を軸、joint_dof_pos[:, i] を角度として回転行列を作成
                axis = self.joint_axis_init[:, i, :].unsqueeze(1)  # (num_envs, 1, 3)
                theta = self.joint_dof_pos[:, i].unsqueeze(1)        # (num_envs, 1)
                R = self.rotation_matrix(axis, theta)                # (num_envs, 3, 3)
                T_motion = self.embed_rotation(R)
            elif self.joint_types[i] == "prismatic":
                # 平行移動の場合：joint_axis_init に沿って joint_dof_pos[:, i] 分移動
                d = self.joint_dof_pos[:, i].unsqueeze(-1)            # (num_envs, 1)
                translation = self.joint_axis_init[:, i, :] * d       # (num_envs, 3)
                T_motion = self.create_translation_matrix(translation)
            else:
                T_motion = torch.eye(4, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)

            # ③ 局所変換行列 T_joint（親リンク→子リンク間の変換）
            self.t_joint[:, i, :, :] = torch.bmm(T_origin, T_motion)

            # ④ グローバル変換行列 T_global（ベースリンク→子リンク間の変換）
            # 再帰的に計算
            reach_base = False
            parent_idx = self.joint_parent_idx[i]
            self.t_global[:, i, :, :] = self.t_joint[:, i, :, :]
            while not reach_base:
                if parent_idx == -1:
                    reach_base = True
                else:
                    self.t_global[:, i, :, :] = torch.bmm(self.t_joint[:, parent_idx, :, :], self.t_global[:, i, :, :])
                    parent_idx = self.joint_parent_idx[parent_idx]

    def calc_tendon_len(self):
        """
        各tendonについて、各viaの連続する2点間のユークリッド距離の和を計算し、self.tendon_lengthsに格納する。
        """
        for t in range(self.num_tendons):
            via_pos = self.tendon_via_pos[t]
            via_pos_diff = via_pos[:, 1:, :] - via_pos[:, :-1, :]
            via_pos_diff_norm = torch.norm(via_pos_diff, dim=-1)
            self.tendon_lengths[:, t] = torch.sum(via_pos_diff_norm, dim=-1)

    def calc_tendon_jacobian(self):
        """
        Batched tendon jacobian の計算  
        各jointとtendonの組に対し、各via対からmoment armを計算し、環境ごとに合計する。
        """
        J = torch.zeros((self.num_envs, self.num_joints, self.num_tendons), dtype=torch.float32, device=self.device)
        
        for i in range(self.num_joints):
            joint_origin = self.joint_origin[:, i, :]
            joint_axis = self.joint_axis[:, i, :]
            for j in range(self.num_tendons):
                # i番目のjointがj番目のtendonに影響を及ぼす場合、moment armを計算
                if self.joint_names[i] in self.affected_joints[j]:
                    via_pos = self.tendon_via_pos[j]
                    # 隣接するvia間のベクトル
                    via_pos_diff = via_pos[:, 1:, :] - via_pos[:, :-1, :]
                    via_pos_diff_unit = via_pos_diff / (torch.norm(via_pos_diff, dim=-1, keepdim=True) + 1e-6)
                    joint_axis_unit = joint_axis / (torch.norm(joint_axis, dim=-1, keepdim=True) + 1e-6)
                    moment_arm = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
                    if self.joint_types[i] == "revolute":
                        # 回転関節の場合：回転軸とワイヤ直線間の符号付き最短距離を各セグメントで計算して合計
                        # 符号付き最短距離は、関節が正方向に回転するときにワイヤが伸びる場合は正, 縮む場合は負
                        for k in range(via_pos_diff.shape[1]):
                            moment_arm += - self.line_line_signed_distance(joint_origin, joint_axis_unit,
                                                                        via_pos[:, k, :],
                                                                        via_pos_diff_unit[:, k, :])
                    elif self.joint_types[i] == "prismatic":
                        # 直動関節の場合：関節軸とワイヤ直線単位ベクトルの内積を各セグメントで計算して合計
                        for k in range(via_pos_diff.shape[1]):
                            moment_arm += torch.sum(joint_axis_unit * via_pos_diff_unit[:, k, :], dim=1)
                    J[:, i, j] = moment_arm
                else:
                    J[:, i, j] = 0.0
        self.tendon_jacobian = J

    def line_line_signed_distance(self, p0, d0, p1, d1):
        """
        2直線間の符号付き最短距離の計算
        Args:
            p0: (batch_size, num_pairs, 3) – 1本目の直線上の点
            d0: (batch_size, num_pairs, 3) – 1本目の直線の方向ベクトル
            p1: (batch_size, num_pairs, 3) – 2本目の直線上の点
            d1: (batch_size, num_pairs, 3) – 2本目の直線の方向ベクトル
        Returns:
            距離: (batch_size, num_pairs) のテンソル
        """
        # 交差ベクトル
        cross = torch.cross(d0, d1, dim=-1)  # (batch_size, num_pairs, 3)
        cross_norm = torch.norm(cross, dim=-1) + 1e-6  # (batch_size, num_pairs)
        diff = p1 - p0  # (batch_size, num_pairs, 3)
        # diff と cross の内積 / ||cross||
        distance = (diff * cross).sum(dim=-1) / cross_norm  # (batch_size, num_pairs)
        return distance
