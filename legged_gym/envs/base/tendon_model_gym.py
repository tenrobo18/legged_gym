import torch
import yaml
import xml.etree.ElementTree as ET

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
                # 各環境に同じ値を設定（タイル展開）
                origin_tensor = torch.tensor(origin_list, dtype=torch.float32, device=device)
                axis_tensor   = torch.tensor(axis_list, dtype=torch.float32, device=device)
                self.joint_origin_init[:, idx, :] = origin_tensor.unsqueeze(0).expand(num_envs, -1)
                self.joint_axis_init[:, idx, :] = axis_tensor.unsqueeze(0).expand(num_envs, -1)
                # 初期状態の値としてコピー
                self.joint_origin[:, idx, :] = self.joint_origin_init[:, idx, :]
                self.joint_axis[:, idx, :] = self.joint_axis_init[:, idx, :]

        # バッチ用のワイヤ（via）の初期化
        # tendon_via_pos はリスト（長さ：num_tendons）、各要素は shape=(num_envs, num_vias, 3) のテンソル
        self.tendon_via_pos = []
        # tendon_via_indices は後から gym など外部から設定するためのリスト（各viaに対応する剛体のインデックス）
        self.tendon_via_indices = []
        for via_names in self.tendon_via_names:
            num_vias = len(via_names)
            self.tendon_via_pos.append(torch.zeros((num_envs, num_vias, 3), dtype=torch.float32, device=device))
            self.tendon_via_indices.append([None] * num_vias)

        self.tendon_lengths = torch.zeros((num_envs, self.num_tendons), dtype=torch.float32, device=device)
        self.tendon_jacobian = torch.zeros((num_envs, self.num_joints, self.num_tendons), dtype=torch.float32, device=device)

    def rotation_matrix(self, axis, theta):
        """
        バッチ対応の Rodrigues 回転公式
        Args:
            axis: (num_envs, 1, 3) tensor (正規化済みであること)
            theta: (num_envs, 1) tensor
        Returns:
            R: (num_envs, 3, 3) 回転行列
        """
        # 正規化
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-6)
        # バッチ用のクロス行列 K を作成
        a_x = axis[..., 0].unsqueeze(-1).unsqueeze(-1)
        a_y = axis[..., 1].unsqueeze(-1).unsqueeze(-1)
        a_z = axis[..., 2].unsqueeze(-1).unsqueeze(-1)
        zero = torch.zeros_like(a_x)
        K = torch.cat([
            torch.cat([zero, -a_z, a_y], dim=-1),
            torch.cat([a_z, zero, -a_x], dim=-1),
            torch.cat([-a_y, a_x, zero], dim=-1)
        ], dim=-2)  # shape: (num_envs, 1, 3, 3)
        I = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0)
        theta = theta.unsqueeze(-1).unsqueeze(-1)  # shape: (num_envs, 1, 1, 1)
        R = I + torch.sin(theta)*K + (1 - torch.cos(theta))*(K @ K)
        # squeeze次元1（ジョイント単位の1）→ shape: (num_envs, 3, 3)
        return R.squeeze(1)

    def update_state(self, dof_pos_tensor, rigid_body_state):
        """
        各環境分の関節角度・変位および剛体状態から、ジョイントとワイヤ内のviaの位置を一括更新する
        Args:
            dof_pos_tensor: (num_envs, num_joints) の tensor。各環境の各ジョイントの値
            rigid_body_state: (num_envs, num_rigid_bodies, 3) の tensor。各環境の剛体位置
        """
        self.joint_dof_pos = dof_pos_tensor
        # [todo] ジョイント更新（各ジョイントごとにバッチ処理）

        # ワイヤ内の各viaについて、外部から設定済みの rigid_body_state のインデックスを使って位置を更新
        for t, via_names in enumerate(self.tendon_via_names):
            num_vias = len(via_names)
            for i in range(num_vias):
                idx = self.tendon_via_indices[t][i]
                # 例: rigid_body_state[:, idx, :] を各環境のvia位置に設定
                self.tendon_via_pos[t][:, i, :] = rigid_body_state[:, idx, 0:3]

        self.calc_tendon_len()
        self.calc_tendon_jacobian()

        print("jacobian: ", self.tendon_jacobian)

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
        各ジョイントとテンドンの組に対し、各via対からmoment armを計算し、環境ごとに合計する。
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
                        for k in range(via_pos_diff.shape[1]):
                            moment_arm += self.line_line_signed_distance(joint_origin, joint_axis_unit,
                                                                        via_pos[:, k, :],
                                                                        via_pos_diff_unit[:, k, :])
                    elif self.joint_types[i] == "prismatic":
                        # 直動関節の場合：関節軸とワイヤ直線単位ベクトルの内積を各セグメントで計算して合計
                        moment_arm = torch.sum(torch.sum(joint_axis_unit * via_pos_diff_unit, dim=-1), dim=-1)
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


# # ============================================
# # 以下は利用例です（実際のyaml, urdfファイルパスおよびrigid_body_stateの設定が必要）
# # ============================================
# if __name__ == "__main__":
#     model = TendonRobotModel()
#     yaml_path = "params.yaml"
#     urdf_path = "model.urdf"
#     model.initialize(yaml_path, urdf_path)
    
#     # 関節角度/変位の更新例（3関節の場合）
#     dof_pos_list = [0.5, -0.3, 0.2]
#     # rigid_body_state例：各剛体の3次元位置（GPU上tensor）
#     rigid_body_state = torch.tensor([[0,0,0],
#                                      [1,0,0],
#                                      [0,1,0],
#                                      [0,0,1],
#                                      [1,1,0],
#                                      [0,1,1]], dtype=torch.float32, device=self.device)
#     model.update_state(dof_pos_list, rigid_body_state)
    
#     tendon_lengths = model.get_tendon_len()
#     tendon_jacobian = model.get_tendon_jacobian()
    
#     # 結果をCPUに戻して表示
#     print("Tendon Lengths:", [l.item() for l in tendon_lengths])
#     print("Tendon Jacobian:\n", tendon_jacobian.cpu().numpy())
