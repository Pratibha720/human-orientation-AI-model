import mediapipe as mp

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh


def get_pose_landmarks(image):
    """Return pose landmarks or None."""
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(image)

        # If mostly non-human → ignore
        if not res.pose_landmarks:
            return None

        # Reject if only shoulders detected (dogs sometimes match)
        visible = sum(1 for lm in res.pose_landmarks.landmark if lm.visibility > 0.6)
        if visible < 8:          # needs at least 8 keypoints visible
            return None

        return res.pose_landmarks


def get_face_landmarks(image):
    """Return (face_landmarks, face_count)."""

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        res = fd.process(image)

        if not res.detections:
            return None, 0

        face_count = len(res.detections)

    # If MULTIPLE faces → return count only
    if face_count != 1:
        return None, face_count

    # Get face mesh for orientation
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        mesh_res = fm.process(image)
        if mesh_res.multi_face_landmarks:
            return mesh_res.multi_face_landmarks[0], 1

    return None, 1
