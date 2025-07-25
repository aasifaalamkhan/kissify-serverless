import runpod
from inference import generate_kissing_video, set_status_callback

def handler(job):
    input_data = job["input"]

    logs = []

    def capture_status(msg):
        logs.append(msg)

    set_status_callback(capture_status)

    try:
        result = generate_kissing_video(input_data)
        return {
            "status": "success",
            "video_url": result["video_url"],
            "log": logs
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "log": logs
        }

runpod.serverless.start({"handler": handler})
