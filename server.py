# server.py

import asyncio
import logging
import numpy as np
import torch
from config import get_model_configs,SAMPLE_RATE
from typing import Union, List,Dict
from fastapi.staticfiles import StaticFiles
from simuleval.data.segments import SpeechSegment, TextSegment, Segment
from simuleval import options
from simuleval.utils.arguments import cli_argument_list
from simuleval.agents.pipeline import TreeAgentPipeline
from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STJointVADAgent

# FastAPI Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request,Form
from fastapi.responses import HTMLResponse

# Constants and Logger Setup

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("server")


class OutputSegments:
    """Manages the output segments."""
    def __init__(self, segments: Union[List[Segment], Segment]):
        if isinstance(segments, Segment):
            segments = [segments]
        self.segments = segments

    @property
    def is_empty(self):
        return all(segment.is_empty for segment in self.segments)

    @property
    def finished(self):
        return all(segment.finished for segment in self.segments)


def build_streaming_system(model_configs, agent_class):
    parser = options.general_parser()
    parser.add_argument("-f", "--f", help="a dummy argument to fool ipython", default="1")
    agent_class.add_args(parser)
    args, _ = parser.parse_known_args(cli_argument_list(model_configs))
    system = agent_class.from_args(args)
    return system


class Transcoder:
    """Transcoder class to handle model inference and audio processing."""
    def __init__(self, model_configs):
        self.agent = build_streaming_system(model_configs, SeamlessStreamingS2STJointVADAgent)
        device = torch.device(model_configs.get("device", "cpu"))
        dtype = torch.float16 if model_configs.get("dtype") == "fp16" else torch.float32
        self.agent.to(device, dtype=dtype)
        self.sample_rate = SAMPLE_RATE
        self.states = self.agent.build_states()
        self.lock = asyncio.Lock()

        # Audio buffer to hold raw audio data
        self.audio_buffer = bytearray()
        self.end_stream = False

        # Queues for different stages of processing
        self.preprocessed_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

    def add_to_buffer(self, audio_data: bytes):
        """Add incoming audio data to the buffer."""
        self.audio_buffer.extend(audio_data)

    def mark_end_of_stream(self):
        """Mark the stream as ended."""
        self.end_stream = True

    async def run(self):
        """Main transcoder loop to process incoming audio."""
        tasks = [
            asyncio.create_task(self.ingest_audio()),
            asyncio.create_task(self.process_audio()),
        ]
        await asyncio.gather(*tasks)

    async def ingest_audio(self):
        """Handle raw audio input, chunking and queuing for processing."""
        CHUNK_SIZE = 1024 * 8  # Set a chunk size, e.g., 8192 bytes

        while not self.end_stream or len(self.audio_buffer) > 0:
            # If there's data in the buffer and it exceeds CHUNK_SIZE, process it
            while len(self.audio_buffer) >= CHUNK_SIZE:
                audio_chunk = self.audio_buffer[:CHUNK_SIZE]
                self.audio_buffer = self.audio_buffer[CHUNK_SIZE:]

                # Convert the audio chunk into a SpeechSegment
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                input_segment = SpeechSegment(
                    content=audio_np,
                    sample_rate=self.sample_rate,
                    tgt_lang=TGT_LANG,
                    finished=False,
                )
                await self.preprocessed_queue.put(input_segment)

            # Flush remaining buffer if stream has ended
            if self.end_stream and len(self.audio_buffer) > 0:
                audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                input_segment = SpeechSegment(
                    content=audio_np,
                    sample_rate=self.sample_rate,
                    tgt_lang=TGT_LANG,
                    finished=True,
                )
                await self.preprocessed_queue.put(input_segment)
                self.audio_buffer = bytearray()  # Clear the buffer
                break

            await asyncio.sleep(0.01)

    async def process_audio(self):
        """Process preprocessed audio segments."""
        while not self.end_stream or not self.preprocessed_queue.empty():
            try:
                input_segment = await asyncio.wait_for(self.preprocessed_queue.get(), timeout=1.0)
                self.preprocessed_queue.task_done()
                async with self.lock:
                    with torch.no_grad():
                        output = await asyncio.get_event_loop().run_in_executor(
                            None, self.agent.pushpop, input_segment, self.states
                        )
                output_segments = OutputSegments(output)
                if not output_segments.is_empty:
                    await self.output_queue.put(output_segments)
                if output_segments.finished:
                    logger.info("[process_audio] Finished processing, resetting states")
                    self.reset_states()
            except asyncio.TimeoutError:
                if self.end_stream and self.preprocessed_queue.empty():
                    logger.info("[process_audio] No more preprocessed data, ending process_audio.")
                    break
            except Exception as e:
                logger.exception(f"[process_audio] Exception: {e}")

    def reset_states(self):
        """Reset model states after finishing a translation."""
        logger.info("[reset_states] Resetting states")
        if isinstance(self.agent, TreeAgentPipeline):
            states_iter = self.states.values()
        else:
            states_iter = self.states
        for state in states_iter:
            state.reset()


# FastAPI application
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/set-language/")
async def set_language(language: str = Form(...)) -> Dict[str, str]:
    """Set the target language and return the selected language."""
    global TGT_LANG
    TGT_LANG = language
    return {"message": f"Target language set to {TGT_LANG}"}



async def send_output(websocket: WebSocket, transcoder: Transcoder):
    """Sends output audio to the client."""
    try:
        while True:
            try:
                output_segments = await asyncio.wait_for(transcoder.output_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if transcoder.end_stream and transcoder.output_queue.empty():
                    logger.info("Ending send_output as end_stream is reached and output queue is empty.")
                    break
                continue

            for segment in output_segments.segments:
                if isinstance(segment, SpeechSegment):
                    translated_bytes = (np.array(segment.content) * 32768).astype(np.int16).tobytes()
                    logger.info("[send_output] Got speech segment, length: %d", len(translated_bytes))
                    await websocket.send_bytes(translated_bytes)
                elif isinstance(segment, TextSegment):
                    logger.info(f"[send_output] Got text segment: {segment.content}")
                    await websocket.send_text(segment.content)
            transcoder.output_queue.task_done()

    except asyncio.CancelledError:
        logger.info("[send_output] Sender task cancelled")
    except Exception as e:
        logger.exception(f"[send_output] Error: {e}")

def cancel_tasks(*tasks):
    """Cancels running tasks."""
    logger.info("Cancelling tasks")
    for task in tasks:
        if task:
            task.cancel()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    transcoder = Transcoder(get_model_configs())
    transcoder_task = asyncio.create_task(transcoder.run())
    sender_task = asyncio.create_task(send_output(websocket, transcoder))
    try:
        while True:
            data = await websocket.receive_bytes()
            transcoder.add_to_buffer(data)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        transcoder.mark_end_of_stream()
        cancel_tasks(transcoder_task, sender_task)
    except Exception as e:
        logger.exception(f"WebSocket Error: {e}")
        transcoder.mark_end_of_stream()
        cancel_tasks(transcoder_task, sender_task)

# Add this endpoint to serve your index.html
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return HTMLResponse(open("static/index.html").read())

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run("server:app", host="0.0.0.0", port=10006, log_level="info")
