let websocket;
        let mediaRecorder;
        let audioContext;
        let audioBufferQueue = [];
        let sourceNode;
        let processor;
        let inputStream;

        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const messagesDiv = document.getElementById('messages');

////

// static/script.js
document.getElementById("languageSelect").addEventListener("change", async function () {
    const languageSelect = document.getElementById("languageSelect");
    const language = languageSelect.value;

    const formData = new FormData();
    formData.append("language", language);

    const response = await fetch("/set-language/", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("result").innerText = result.message;
});





////


        startButton.onclick = async function() {
            startButton.disabled = true;
            stopButton.disabled = false;
            // Initialize WebSocket
            websocket = new WebSocket('ws://localhost:10006/ws');

            websocket.binaryType = 'arraybuffer';

            websocket.onopen = function() {
                console.log('WebSocket connection opened');
                startRecording();
            };

            websocket.onmessage = function(event) {
                if (typeof event.data === 'string') {
                    // Received text message
                    const text = event.data;
                    messagesDiv.innerHTML += `<p>${text}</p>`;
                } else {
                    // Received binary data (audio)
                    const arrayBuffer = event.data;
                    playAudioBuffer(arrayBuffer);
                }
            };

            websocket.onclose = function() {
                console.log('WebSocket connection closed');
                stopRecording();
            };

            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        };

        stopButton.onclick = function() {
            stopButton.disabled = true;
            startButton.disabled = false;
            if (websocket) {
                websocket.close();
            }
        };

        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                inputStream = stream;
                const source = audioContext.createMediaStreamSource(stream);

                processor = audioContext.createScriptProcessor(4096, 1, 1);
                processor.onaudioprocess = function(e) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const int16Data = convertFloat32ToInt16(inputData);
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(int16Data.buffer);
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);
            }).catch(function(err) {
                console.error('Error accessing microphone:', err);
            });
        }

        function stopRecording() {
            if (processor && audioContext) {
                processor.disconnect();
                processor.onaudioprocess = null;
                processor = null;
            }
            if (inputStream) {
                inputStream.getTracks().forEach(track => track.stop());
                inputStream = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
        }

        function convertFloat32ToInt16(buffer) {
            let l = buffer.length;
            const result = new Int16Array(l);
            for (let i = 0; i < l; i++) {
                let s = Math.max(-1, Math.min(1, buffer[i]));
                result[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            return result;
        }

        function playAudioBuffer(arrayBuffer) {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            }

            const int16Array = new Int16Array(arrayBuffer);
            const float32Array = new Float32Array(int16Array.length);

            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768;
            }

            const audioBuffer = audioContext.createBuffer(1, float32Array.length, 16000);
            audioBuffer.getChannelData(0).set(float32Array);

            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start();
        }