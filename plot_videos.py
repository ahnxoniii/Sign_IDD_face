import os
import cv2
import math
import torch
import numpy as np
from dtw import dtw
from constants import PAD_TOKEN

NUM_JOINTS = 177
TRG_SIZE = NUM_JOINTS * 3  # 531

# This is the format of the 3D data, outputted from the Inverse Kinematics model

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
    skeleton_2d = (
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

        # right hand
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),

        (7, 12),
        (12, 13),
        (13, 14),
        (14, 15),

        (7, 16),
        (16, 17),
        (17, 18),
        (18, 19),

        (7, 20),
        (20, 21),
        (21, 22),
        (22, 23),

        (7, 24),
        (24, 25),
        (25, 26),
        (26, 27),

        # left hand
        (3, 28),
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),

        (28, 33),
        (33, 34),
        (34, 35),
        (35, 36),

        (28, 37),
        (37, 38),
        (38, 39),
        (39, 40),

        (28, 41),
        (41, 42),
        (42, 43),
        (43, 44),

        (28, 45),
        (45, 46),
        (46, 47),
        (47, 48),

        *face_skeletons_cvpr,
    )

    # plot_video.py는 (start, end, bone_type) 3개짜리 필요
    return tuple((p, c, 0) for p, c in skeleton_2d)


# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None):
    # Create video template
    FPS = (25 // skip_frames)
    # ipdb.set_trace()
    video_file = file_path + "/{}.mp4".format(sequence_ID.split(".")[0])
    video_path, video_name = os.path.split(video_file)
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (650, 650), True)
    elif references is not None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (1300, 650), True)  # Long

    num_frames = 0



    for (j, frame_joints) in enumerate(joints):

        # Reached padding
        if PAD_TOKEN in frame_joints.astype('str').tolist():
            continue

        # Initialise frame of white
        frame = np.ones((650, 650, 3), np.uint8) * 255

        # Cut off the percent_tok, multiply by 3 to restore joint size
        # TODO - Remove the *3 if the joints weren't divided by 3 in data creation
        frame_joints = frame_joints[:-1] * 3

        # Reduce the frame joints down to 2D for visualisation - Frame joints 2d shape is (48,2)
        frame_joints_2d = np.reshape(frame_joints, (NUM_JOINTS, 3))[:, :2]
        # Draw the frame given 2D joints
        draw_frame_2D(frame, frame_joints_2d)

        cv2.putText(frame, "Predicted Sign Pose", (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            # Cut off the percent_tok and multiply each joint by 3 (as was reduced in training files)
            ref_joints = ref_joints[:-1] * 3

            # Reduce the frame joints down to 2D- Frame joints 2d shape is (48,2)
            ref_joints_2d = np.reshape(ref_joints, (NUM_JOINTS, 3))[:, :2]

            # Draw these joints on the frame
            draw_frame_2D(ref_frame, ref_joints_2d)

            cv2.putText(ref_frame, "Ground Truth Pose", (190, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

            frame = np.concatenate((frame, ref_frame), axis=1)

            sequence_ID_write = "Sequence ID: " + sequence_ID.split("/")[-1]
            cv2.putText(frame, sequence_ID_write, (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)
        # Write the video frame
        video.write(frame)
        num_frames += 1
    # Release the video
    video.release()


# Draw a line between two points, if they are positive points
def draw_line(im, joint1, joint2, c=(0, 0, 255),t=1, width=3):
    thresh = -100
    if joint1[0] > thresh and  joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:

        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))

        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)

        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))

        cv2.ellipse(im, center, (width,length), -angle,0.0,360.0, c, -1)

# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(frame, joints):
    # Line to be between the stacked
    draw_line(frame, [1, 650], [1, 1], c=(0,0,0), t=1, width=1)
    # Give an offset to center the skeleton around
    offset = [350, 250]

    # Get the skeleton structure details of each bone, and size
    skeleton = getSkeletalModelStructure()
    skeleton = np.array(skeleton)

    number = skeleton.shape[0]

    # Increase the size and position of the joints
    joints = joints * 10 * 12 * 2
    joints = joints + np.ones((NUM_JOINTS, 2)) * offset

    # Loop through each of the bone structures, and plot the bone
    for j in range(number):

        c = get_bone_colour(skeleton,j)

        draw_line(frame, [joints[skeleton[j, 0]][0], joints[skeleton[j, 0]][1]],
                  [joints[skeleton[j, 1]][0], joints[skeleton[j, 1]][1]], c=c, t=1, width=1)

# get bone colour given index
def get_bone_colour(skeleton,j):

    return (0, 0, 0)

# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq):

    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # Cut the reference down to the max count value
    _ , ref_max_idx = torch.max(ref_seq[:, -1], 0)
    if ref_max_idx == 0: ref_max_idx += 1
    # Cut down frames by counter
    ref_seq = ref_seq[:ref_max_idx,:].cpu().numpy()

    # Cut the hypothesis down to the max count value
    _, hyp_max_idx = torch.max(pred_seq[:, -1], 0)
    if hyp_max_idx == 0: hyp_max_idx += 1
    # Cut down frames by counter
    pred_seq = pred_seq[:hyp_max_idx,:].cpu().numpy()
    #pred_seq = pred_seq[:ref_max_idx, :].cpu().numpy()
    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq[:,:-1], pred_seq[:,:-1], dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]

    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    # j tracks the position in the reference sequence
    j = 0
    skips = 0
    squeeze_frames = []
    for (i, pred_num) in enumerate(path[0]):

        if i == len(path[0]) - 1:
            break

        if path[1][i] == path[1][i + 1]:
            skips += 1

        # If a double coming up
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        # Just finished a double
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d

# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame