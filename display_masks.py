import cv2
import numpy as np

def main():
    reference = cv2.imread("test_norm_mask.png", cv2.IMREAD_GRAYSCALE)
    moving = cv2.imread("test_flir_mask.png", cv2.IMREAD_GRAYSCALE)
    homography = np.load("test_homography.npy")
    opacity = 0.6

    assert reference is not None
    assert moving is not None
    assert homography is not None

    rw = reference.shape[1]
    rh = reference.shape[0]

    aligned = cv2.warpPerspective(moving, homography, (rw, rh))

    overlay = np.zeros((rh, rw, 3), dtype=np.float32)
    overlay[..., 0] = (opacity * (reference / 255.0))
    overlay[..., 2] = (opacity * (aligned / 255.0))

    cv2.imshow("Aligned Masks", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()