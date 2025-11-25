# 1. FULL BODY DETECTION
def is_full_body(pose_landmarks):
    """Check if enough lower-body points are visible."""
    if pose_landmarks is None:
        return False

    try:
        body_points = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles
        visible = sum(
            1 for i in body_points
            if pose_landmarks.landmark[i].visibility > 0.55
        )
        return visible >= 4
    except:
        return False



# 2. FACE ORIENTATION (USED ONLY IF FACE CLEAR)

def face_orientation(face_landmarks, width):
    """Front/Left/Right using FaceMesh."""
    if face_landmarks is None:
        return None

    try:
        nose = face_landmarks.landmark[1]
        left_cheek = face_landmarks.landmark[234]
        right_cheek = face_landmarks.landmark[454]

        nx = nose.x
        lx = left_cheek.x
        rx = right_cheek.x

        center = (lx + rx) / 2
        margin = 0.02

        if abs(nx - center) < margin:
            return "Front"

        # Nose closer to LEFT side → facing RIGHT
        if nx < center:
            return "Right"

        # Nose closer to RIGHT side → facing LEFT
        return "Left"

    except:
        return None



# 3. POSE-BASED ORIENTATION (MOST RELIABLE)

def pose_left_right(pose_landmarks):
    """
    Accurate left/right detection using:
    nose + shoulders midpoint.
    Works even when face is side-profile.
    """
    try:
        nose_x = pose_landmarks.landmark[0].x
        ls = pose_landmarks.landmark[11].x
        rs = pose_landmarks.landmark[12].x

        mid = (ls + rs) / 2  # shoulder midpoint

        # near midpoint → front
        if abs(nose_x - mid) < 0.02:
            return "Front"

        # nose right of midpoint → person facing LEFT
        if nose_x > mid:
            return "Left"

        # nose left of midpoint → person facing RIGHT
        return "Right"

    except:
        return None



# 4. FINAL ORIENTATION DECISION LOGIC

def combine_orientation(pose_landmarks, face_landmarks, face_count, width):
    """
    Combines:
    - Full Body
    - Face orientation (if reliable)
    - Pose-based fallback (most accurate for side views)
    """

    # ---- 1. Multiple people → N/A ----
    if face_count > 1:
        return "N/A"

    # ---- 2. No human at all ----
    if pose_landmarks is None and face_landmarks is None:
        return "N/A"

    # ---- 3. Full body → highest priority ----
    if pose_landmarks and is_full_body(pose_landmarks):
        return "Full Body"

    # ---- 4. Pose-based left/right (BEST SIDE VIEW DETECTION) ----
    if pose_landmarks:
        pose_dir = pose_left_right(pose_landmarks)
        if pose_dir:
            return pose_dir

    # ---- 5. Face orientation (secondary) ----
    if face_landmarks and face_count == 1:
        face_dir = face_orientation(face_landmarks, width)
        if face_dir:
            return face_dir


    # 6. Default
    return "N/A"


