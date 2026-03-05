import os
import base64
import tempfile
import traceback
import numpy as np
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
_CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "gemini-3-pro-preview")


def _get_caption_client():
    return OpenAI(base_url=OPENAI_BASE_URL)


def _extract_segment_as_base64(video, start, end):
    """Extract a video subclip and return it as a base64-encoded mp4 string."""
    subclip = video.subclip(start, end)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subclip.write_videofile(tmp_path, codec="libx264", verbose=False, logger=None)
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        os.unlink(tmp_path)


def _caption_with_api(client, video_base64, query):
    """Call AIHubMix Gemini API with a video segment and a text query."""
    content = [
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
        },
        {"type": "text", "text": query},
    ]

    response = client.chat.completions.create(
        model=_CAPTION_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def segment_caption(
    video_name,
    video_path,
    segment_index2name,
    transcripts,
    segment_times_info,
    caption_result,
    error_queue,
):
    try:
        client = _get_caption_client()

        with VideoFileClip(video_path) as video:
            for index in tqdm(
                segment_index2name, desc=f"Captioning Video {video_name}"
            ):
                start, end = segment_times_info[index]["timestamp"]
                video_base64 = _extract_segment_as_base64(video, start, end)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                result = _caption_with_api(client, video_base64, query)
                caption_result[index] = result.replace("\n", "").replace(
                    "<|endoftext|>", ""
                )
    except Exception as e:
        print("Error:")
        print(traceback.format_exc())
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError


def merge_segment_information(
    segment_index2name, segment_times_info, transcripts, captions
):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = "-".join(segment_name.split("-")[-2:])
        inserting_segments[index]["content"] = (
            f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        )
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index][
            "frame_times"
        ].tolist()
    return inserting_segments


def retrieved_segment_caption(
    refine_knowledge,
    retrieved_segments,
    video_path_db,
    video_segments,
    num_sampled_frames,
):
    client = _get_caption_client()

    caption_result = {}
    for this_segment in tqdm(
        retrieved_segments, desc="Captioning Segments for Given Query"
    ):
        video_name = "_".join(this_segment.split("_")[:-1])
        index = this_segment.split("_")[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split("-")
        start, end = eval(timestamp[0]), eval(timestamp[1])

        with VideoFileClip(video_path) as video:
            video_base64 = _extract_segment_as_base64(video, start, end)

        segment_transcript = video_segments._data[video_name][index]["transcript"]
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        result = _caption_with_api(client, video_base64, query)
        this_caption = result.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = (
            f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        )

    return caption_result
