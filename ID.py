# coding: utf-8
import torch

from helpers import getSkeletalModelStructure

def ID(trg):
    
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 177, 3) # 150 -> 50, 3으로 형태 변경
    trg_list = trg_reshaped.split(1, dim=2) # 50개 관절을 다 분리해서 리스트로 만듬
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list] # 2번째 차원이 1로 다 되버리기 때문에 없애부림
    skeletons = getSkeletalModelStructure() # 스켈레톤 구조 가꼬 와서 
    trg_reshaped_list = [] # 스켈레톤 엣지 계산된 결과를 넣음
    for skeleton in skeletons:
        # 두 관절 사이 거리 재기 len
        Skeleton_length = torch.norm(trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]], p=2, dim=2, keepdim=True)
        # 방향 벡터 계산 dx, dy, dz
        Skeleton_direct = (trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]]) / (Skeleton_length+torch.finfo(Skeleton_length.dtype).tiny)
        # 2번째 차원으로 concat 7차원이 됨 [x_b, y_b, z_b, length, dx, dy, dz]
        trg_reshaped_list.append(torch.cat((trg_list_squeeze[skeleton[1]], Skeleton_length, Skeleton_direct), dim=2))
    #trg_reshaped_list을 하나의 텐서로 만들고 형태 변경
    trg_super = torch.stack(trg_reshaped_list, dim=-1).reshape(trg.shape[0],trg.shape[1],177*7)

    return trg_super

# face joints: 50~117
# local fa68 index -> global index
#  0~16  -> 50~66   jaw 턱
# 17~21  -> 67~71   left eyebrow
# 22~26  -> 72~76   right eyebrow
# 27~30  -> 77~80   nose bridge / tip
# 31~35  -> 81~85   lower nose
# 36~41  -> 86~91   left eye
# 42~47  -> 92~97   right eye
# 48~59  -> 98~109  outer mouth
# 60~67  -> 110~117 inner mouth

# (parent, child) (1, 77) direction = parent - child

face_skeletons = (
    # ----- new face root at nose 29 -----
    (1, 79),      # body joint 1 -> face global 79  (fa local 29)

  # ----- nose bridge / nose -----
    (79, 78),     # fa 29 -> 28
    (78, 77),     # fa 28 -> 27

    (79, 80),     # fa 29 -> 30
    (80, 83),     # fa 30 -> 33
    (83, 82),     # fa 33 -> 32
    (82, 81),     # fa 32 -> 31
    (83, 84),     # fa 33 -> 34
    (84, 85),     # fa 34 -> 35

    # ----- left eyebrow (fa 17~21 / global 67~71) -----
    (77, 71),     # fa 27 -> fa 21
    (71, 70),     # fa 21 -> fa 20
    (70, 69),     # fa 20 -> fa 19
    (69, 68),     # fa 19 -> fa 18
    (68, 67),     # fa 18 -> fa 17

    # ----- right eyebrow (fa 22~26 / global 72~76) -----
    (77, 72),     # fa 27 -> fa 22
    (72, 73),     # fa 22 -> fa 23
    (73, 74),     # fa 23 -> fa 24
    (74, 75),     # fa 24 -> fa 25
    (75, 76),     # fa 25 -> fa 26

    # ----- left eye (fa 36~41 / global 86~91) -----
    (71, 89),     # face global 71 (fa 21) -> face global 89 (fa 39)  # inner brow -> inner eye
    (89, 88),     # fa 39 -> 38
    (88, 87),     # fa 38 -> 37
    (87, 86),     # fa 37 -> 36
    (89, 90),     # fa 39 -> 40
    (90, 91),     # fa 40 -> 41

    # ----- right eye (fa 42~47 / global 92~97) -----
    (72, 92),     # face global 72 (fa 22) -> face global 92 (fa 42)  # inner brow -> inner eye
    (92, 93),     # fa 42 -> 43
    (93, 94),     # fa 43 -> 44
    (94, 95),     # fa 44 -> 45
    (92, 97),     # fa 42 -> 47
    (97, 96),     # fa 47 -> 46

    # ----- outer mouth (fa 48~59 / global 98~109) -----
    (83, 101),    # fa 33 -> fa 51

    # left branch
    (101, 100),   # fa 51 -> 50
    (100, 99),    # fa 50 -> 49
    (99, 98),     # fa 49 -> 48

    # right branch
    (101, 102),   # fa 51 -> 52
    (102, 103),   # fa 52 -> 53
    (103, 104),   # fa 53 -> 54

    # lower branch
    (101, 107),   # fa 51 -> 57
    (107, 106),   # fa 57 -> 56
    (106, 105),   # fa 56 -> 55
    (107, 108),   # fa 57 -> 58
    (108, 109),   # fa 58 -> 59

    # ----- inner mouth (fa 60~67 / global 110~117) -----
    (101, 112),   # face global 101 (fa 51) -> face global 112 (fa 62)
    (112, 111),   # face global 112 (fa 62) -> face global 111 (fa 61)
    (111, 110),   # face global 111 (fa 61) -> face global 110 (fa 60)
    (112, 113),   # face global 112 (fa 62) -> face global 113 (fa 63)
    (113, 114),   # face global 113 (fa 63) -> face global 114 (fa 64)
    (114, 115),   # face global 114 (fa 64) -> face global 115 (fa 65)
    (115, 116),   # face global 115 (fa 65) -> face global 116 (fa 66)
    (116, 117),   # face global 116 (fa 66) -> face global 117 (fa 67)

    # ----- jaw (fa 0~16 / global 50~66) -----
    (107, 58),    # face global 107 (fa 57) -> face global 58 (fa 8)

    (58, 57),     # fa 8 -> 7
    (57, 56),     # fa 7 -> 6
    (56, 55),     # fa 6 -> 5
    (55, 54),     # fa 5 -> 4
    (54, 53),     # fa 4 -> 3
    (53, 52),     # fa 3 -> 2
    (52, 51),     # fa 2 -> 1
    (51, 50),     # fa 1 -> 0

    (58, 59),     # fa 8 -> 9
    (59, 60),     # fa 9 -> 10
    (60, 61),     # fa 10 -> 11
    (61, 62),     # fa 11 -> 12
    (62, 63),     # fa 12 -> 13
    (63, 64),     # fa 13 -> 14
    (64, 65),     # fa 14 -> 15
    (65, 66),     # fa 15 -> 16
)

face_skeletons_cvpr = (
    # =====================================================
    # Face outline independent
    # root: 173
    # 173에서 양갈래로 퍼짐
    # =====================================================

    (173, 173),

    # branch 1: 173 -> ... -> 67
    (173, 153),
    (153, 109),
    (109, 108),
    (108, 135),
    (135, 98),
    (98, 97),
    (97, 138),
    (138, 75),
    (75, 74),
    (74, 136),
    (136, 134),
    (134, 133),
    (133, 144),
    (144, 176),
    (176, 117),
    (117, 116),
    (116, 131),
    (131, 67),

    # branch 2: 173 -> ... opposite outline
    (173, 125),
    (125, 126),
    (126, 95),
    (95, 96),
    (96, 149),
    (149, 150),
    (150, 168),
    (168, 113),
    (113, 114),
    (114, 115),
    (115, 170),
    (170, 132),
    (132, 87),
    (87, 88),
    (88, 92),
    (92, 93),
    (93, 66),

    # =====================================================
    # Right eyebrow independent
    # root: 152
    # =====================================================
    (152, 152),

    # upper
    (152, 151),
    (151, 77),
    (77, 76),
    (76, 146),

    # lower
    (152, 121),
    (121, 120),
    (120, 143),
    (143, 142),
    (142, 160),

    # =====================================================
    # Left eyebrow independent
    # root: 69
    # =====================================================
    (69, 69),

    # upper
    (69, 68),
    (68, 145),
    (145, 56),
    (56, 55),

    # lower
    (69, 175),
    (175, 154),
    (154, 111),
    (111, 110),
    (110, 157),

    # =====================================================
    # Right eye independent
    # root: 59
    # =====================================================
    (59, 59),

    # upper/outward branch
    (59, 169),
    (169, 130),
    (130, 101),
    (101, 100),
    (100, 137),
    (137, 71),
    (71, 70),
    (70, 99),

    # lower/outward branch
    (59, 52),
    (52, 51),
    (51, 158),
    (158, 84),
    (84, 65),
    (65, 64),
    (64, 155),

    # =====================================================
    # Left eye independent
    # root: 119
    # =====================================================
    (119, 119),

    # upper/outward branch
    (119, 89),
    (89, 73),
    (73, 72),
    (72, 148),
    (148, 107),
    (107, 106),
    (106, 159),
    (159, 102),

    # lower/outward branch
    (119, 118),
    (118, 156),
    (156, 147),
    (147, 63),
    (63, 62),
    (62, 82),
    (82, 103),

    # =====================================================
    # Outer mouth independent
    # root: 81 = upper lip center
    # =====================================================
    (81, 81),

    # one side: 81 -> ... -> 127
    (81, 80),
    (80, 124),
    (124, 123),
    (123, 105),
    (105, 104),
    (104, 163),
    (163, 172),
    (172, 140),
    (140, 141),
    (141, 127),

    # other side: 81 -> ... -> 128
    (81, 161),
    (161, 162),
    (162, 85),
    (85, 86),
    (86, 79),
    (79, 78),
    (78, 112),
    (112, 167),
    (167, 128),

    # =====================================================
    # Inner mouth independent
    # root: 50 = inner upper lip center
    # =====================================================
    (50, 50),

    # one side: 50 -> ... -> 58
    (50, 83),
    (83, 53),
    (53, 54),
    (54, 129),
    (129, 94),
    (94, 91),
    (91, 90),
    (90, 164),
    (164, 122),
    (122, 58),

    # other side: 50 -> ... -> 57
    (50, 49),
    (49, 61),
    (61, 60),
    (60, 166),
    (166, 165),
    (165, 171),
    (171, 174),
    (174, 139),
    (139, 57),
)

def getSkeletalModelStructure():
    return (
        # Neck root
        (0, 0),   

        # left arm
        (0, 1),
        (1, 2),
        (2, 3),

        # right arm
        (0, 4),
        (4, 5),
        (5, 6),

        # right 손목 -> 연결
        (6, 7),   

        # right 손 엄지
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),

        # right 손 검지
        (7, 12),
        (12, 13),
        (13, 14),
        (14, 15),

        # right 손 중지
        (7, 16),
        (16, 17),
        (17, 18),
        (18, 19),

        # right 손 약지
        (7, 20),
        (20, 21),
        (21, 22),
        (22, 23),

        # right 손 소지
        (7, 24),
        (24, 25),
        (25, 26),
        (26, 27),


        # left hand: connected from left wrist 3
        (3, 28),

        #left 손 엄지
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),

        #left 손 검지
        (28, 33),
        (33, 34),
        (34, 35),
        (35, 36),

        #left 손 중지
        (28, 37),
        (37, 38),
        (38, 39),
        (39, 40),

        #left 손 약지
        (28, 41),
        (41, 42),
        (42, 43),
        (43, 44),

        #left 손 소지
        (28, 45),
        (45, 46),
        (46, 47),
        (47, 48),
        
        *face_skeletons_cvpr
    )