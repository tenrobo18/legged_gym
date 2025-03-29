import torch
import yaml
import xml.etree.ElementTree as ET

class TendonRobotModel:
    class Joint:
        def __init__(self, name, joint_type, origin_init, axis_init, device):
            self.name = name
            self.type = joint_type  # "revolute" または "prismatic"
            self.device = device
            # 初期位置・軸をGPU上のtensorとして保持
            self.origin_init = torch.tensor(origin_init, dtype=torch.float32, device=self.device)
            self.axis_init = torch.tensor(axis_init, dtype=torch.float32, device=self.device)
            # 初期状態ではorigin, axisはinit値と同じ
            self.origin = self.origin_init.clone()
            self.axis = self.axis_init.clone()
            # 関節角度（または変位）; スカラーtensor
            self.dof_pos = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    class Via:
        def __init__(self, name, device, via_rigid_body_indices=None):
            self.name = name
            self.device = device
            # viaの位置は3次元ベクトル
            self.pos = torch.zeros(3, dtype=torch.float32, device=self.device)
            # rigid_body_state中での親リンクのindexリスト
            self.via_rigid_body_indices = via_rigid_body_indices if via_rigid_body_indices is not None else []

    class Tendon:
        def __init__(self, name, via_list_info, device):
            self.name = name
            self.device = device
            # ワイヤ長（scalarのGPU tensor）
            self.l = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            # 各via情報（yaml中のViaPoints情報をもとに、nameなどを初期化）
            self.vias = []
            # via_list_infoは各viaの辞書（例：{"ParentLink": "point_base_0_0"}）のリスト
            for via_info in via_list_info:
                via = TendonRobotModel.Via(name=via_info["ParentLink"], device=self.device, via_rigid_body_indices=[])
                self.vias.append(via)

    def __init__(self, yaml_path, urdf_path, device):
        """
        1. params.yamlを読み込み、JointList（ジョイント名）およびTendonList（各テンドンのViaPointsのname）を取得
        2. URDFを読み込み、各jointについて同一の名前を持つjointのorigin, axis, typeをコピーして上書きする
        """
        self.joints = []   # Jointのリスト
        self.tendons = []  # Tendonのリスト

        self.device = device

        # YAMLの読み込み
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        joint_names = params["JointList"]
        
        # YAML上での初期値（仮の値）を設定
        # ※URDFの情報で上書きされるので、ここではダミー値でOKです。
        for name in joint_names:
            # 仮のaxis_init, origin_init
            origin_init = [0, 0, 0]
            axis_init = [0, 0, 1]
            # 一旦"revolute"として生成；URDFで上書きします
            joint = TendonRobotModel.Joint(name, "revolute", origin_init, axis_init, self.device)
            self.joints.append(joint)
        
        # TendonListの読み込み
        tendon_list = params["TendonList"]
        for tendon_info in tendon_list:
            tendon_id = tendon_info["TendonId"]
            via_points = tendon_info["ViaPoints"]
            tendon = TendonRobotModel.Tendon(name=f"tendon_{tendon_id}", via_list_info=via_points, device=self.device)
            self.tendons.append(tendon)

        # URDFの読み込み：各<joint>要素からname, type, origin, axisを取得
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint_elem in root.findall("joint"):
            jname = joint_elem.get("name")
            jtype = joint_elem.get("type")
            mapped_type = "revolute" if jtype == "revolute" else ("prismatic" if jtype == "prismatic" else jtype)
            
            # origin要素のパース（xyz属性）
            origin_elem = joint_elem.find("origin")
            if origin_elem is not None:
                origin_str = origin_elem.get("xyz", "0 0 0")
                origin_list = [float(x) for x in origin_str.split()]
            else:
                origin_list = [0, 0, 0]
            
            # axis要素のパース（xyz属性）
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_str = axis_elem.get("xyz", "0 0 1")
                axis_list = [float(x) for x in axis_str.split()]
            else:
                axis_list = [0, 0, 1]
            
            # 対応するjointの更新
            for joint in self.joints:
                if joint.name == jname:
                    joint.type = mapped_type
                    joint.origin_init = torch.tensor(origin_list, dtype=torch.float32, device=self.device)
                    joint.axis_init = torch.tensor(axis_list, dtype=torch.float32, device=self.device)
                    # 初期状態のorigin, axisを更新
                    joint.origin = joint.origin_init.clone()
                    joint.axis = joint.axis_init.clone()
                    break

#     def update_state(self, dof_pos_list, rigid_body_state):
#         """
#         1. 各jointのdof_posおよび各tendon内のviaのposを更新
#         2. 各jointについて、dof_posとorigin_init, axis_initから計算した回転行列を用いてorigin, axisを更新
#            ※revoluteの場合は回転行列 prismaticの場合は平行移動として更新
#         rigid_body_state:
#             GPU上のtensor (num_rigid_bodies x 3) で、各剛体の位置情報が格納されているものとする
#         """
#         # 各ジョイントの状態更新
#         #[todo]

#         # 各テンドンのviaの位置更新
#         for tendon in self.tendons:
#             for via in tendon.vias:
#                 parts = via.name.split('_')
#                 try:
#                     idx = int(parts[2])
#                 except:
#                     idx = 0
#                 via.via_rigid_body_indices = [idx]
#                 via.pos = rigid_body_state[idx]

#     def rotation_matrix(self, axis, theta):
#         """
#         Rodriguesの回転公式に基づき、与えられた軸(axis)周りの回転角(theta)の回転行列を計算する。
#         """
#         axis = axis / (torch.norm(axis) + 1e-6)
#         K = torch.tensor([[0, -axis[2], axis[1]],
#                           [axis[2], 0, -axis[0]],
#                           [-axis[1], axis[0], 0]], dtype=torch.float32, device=self.device)
#         I = torch.eye(3, device=self.device)
#         R = I + torch.sin(theta)*K + (1 - torch.cos(theta))*(K @ K)
#         return R

#     def get_tendon_len(self):
#         """
#         各tendonについて、各viaの連続する2点間のユークリッド距離の和を計算し、tendon.lに格納する。
#         戻り値は各テンドンの長さを格納したGPU tensorのリスト。
#         """
#         tendon_lengths = []
#         for tendon in self.tendons:
#             length = torch.tensor(0.0, dtype=torch.float32, device=self.device)
#             vias = tendon.vias
#             for i in range(len(vias)-1):
#                 diff = vias[i+1].pos - vias[i].pos
#                 length = length + torch.norm(diff)
#             tendon.l = length
#             tendon_lengths.append(length)
#         return tendon_lengths

#     def get_tendon_jacobian(self):
#         """
#         tendon_jacobianは shape=(num_joints, num_tendons) の行列とする。
#         各ジョイントとテンドンの組に対し、jointの種類に応じたmoment armを各via対について合計し、
#         tendon_jacobian[i, j]に設定する。
#         """
#         num_joints = len(self.joints)
#         num_tendons = len(self.tendons)
#         J = torch.zeros((num_joints, num_tendons), dtype=torch.float32, device=self.device)
#         for i, joint in enumerate(self.joints):
#             for j, tendon in enumerate(self.tendons):
#                 moment_arm = torch.tensor(0.0, dtype=torch.float32, device=self.device)
#                 vias = tendon.vias
#                 for k in range(len(vias)-1):
#                     p1 = vias[k].pos
#                     p2 = vias[k+1].pos
#                     segment = p2 - p1
#                     if joint.type == "revolute":
#                         d = self.line_line_signed_distance(joint.origin, joint.axis, p1, segment)
#                         moment_arm = moment_arm + d
#                     elif joint.type == "prismatic":
#                         seg_norm = torch.norm(segment)
#                         if seg_norm > 1e-6:
#                             seg_unit = segment / seg_norm
#                             moment_arm = moment_arm + torch.dot(seg_unit, joint.axis / (torch.norm(joint.axis)+1e-6))
#                 J[i, j] = moment_arm
#         return J

#     def line_line_signed_distance(self, p0, d0, p1, d1):
#         """
#         2直線間の符号付き最短距離を計算する補助関数
#         直線1: p0 + t*d0
#         直線2: p1 + s*d1
#         """
#         d1_unit = d1 / (torch.norm(d1) + 1e-6)
#         cross = torch.cross(d0, d1_unit)
#         cross_norm = torch.norm(cross) + 1e-6
#         distance = torch.dot((p1 - p0), cross) / cross_norm
#         return distance

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
