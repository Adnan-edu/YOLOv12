from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker

def main():
    # Read the input video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players
    player_tracker = PlayerTracker(model_path="yolo12n.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detection.pkl")

    # Detect Tennis Ball
    ball_tracker = BallTracker(model_path="models/tennis_ball_best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detection.pkl") # Saving the output of this frame in pkl format. 

    # Draw output
    # Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    # Draw Tennis Ball Bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Save Video
    save_video(output_video_frames, "output_videos/output.avi")


if __name__ == "__main__":
    main()
