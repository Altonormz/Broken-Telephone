// Initialize variables
const numCharactersInput = document.getElementById('numCharacters');
const startAudioBtn = document.getElementById('startAudioBtn');
const stopAudioBtn = document.getElementById('stopAudioBtn');
const loadingScreen = document.getElementById('loadingScreen');
const progressDiv = document.getElementById('progress');
const finalTextP = document.getElementById('finalText');
const transcriptionList = document.getElementById('transcriptionList');

let websocket;
let mediaRecorder;
let audioChunks = [];

function createLoadingCharacters(characterImages) {
    loadingScreen.innerHTML = '';
    characterImages.forEach((characterImage, i) => {
        const charDiv = document.createElement('div');
        charDiv.classList.add('character');

        // Add face SVG
        const faceDiv = document.createElement('div');
        faceDiv.classList.add('face');
        const faceImg = document.createElement('img');
        faceImg.src = `/static/images/${characterImage}`;
        faceImg.alt = `Character ${i + 1}`;
        faceDiv.appendChild(faceImg);
        charDiv.appendChild(faceDiv);

        // Create text bubble
        const textBubble = document.createElement('div');
        textBubble.classList.add('text-bubble');
        charDiv.appendChild(textBubble);

        loadingScreen.appendChild(charDiv);
    });
}

function updateLoadingCharacters(currentCharIndex, currentText) {
    const chars = document.querySelectorAll('.character');

    chars.forEach((char, index) => {
        if (index === currentCharIndex) {
            char.classList.add('highlight');
            const textBubble = char.querySelector('.text-bubble');
            textBubble.innerText = currentText;
        } else {
            char.classList.remove('highlight');
            const textBubble = char.querySelector('.text-bubble');
            textBubble.innerText = '';
        }
    });
}

function startProcess(data) {
    finalTextP.innerText = '';
    progressDiv.innerText = 'Processing...';

    websocket = new WebSocket(`${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/process`);
    websocket.binaryType = 'arraybuffer';

    websocket.onopen = () => {
        websocket.send(JSON.stringify(data));
        // Send audio chunks
        audioChunks.forEach(chunk => {
            // Read the Blob as an ArrayBuffer
            const reader = new FileReader();
            reader.onload = () => {
                const arrayBuffer = reader.result;
                websocket.send(arrayBuffer);
            };
            reader.readAsArrayBuffer(chunk);
        });
        // Wait until all chunks are sent before sending 'END'
        Promise.all(audioChunks.map(chunk => new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = () => {
                websocket.send(reader.result);
                resolve();
            };
            reader.readAsArrayBuffer(chunk);
        }))).then(() => {
            websocket.send('END'); // Indicate end of audio data
        });
    };

    websocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.event === 'init') {
            const characterImages = message.characters;
            createLoadingCharacters(characterImages);
        } else if (message.event === 'play_audio') {
            playAudio(message.audio_url, message.volume);
        } else if (message.event === 'progress') {
            progressDiv.innerText = `Iteration ${message.iteration}`;
            updateLoadingCharacters(message.character_index, message.text);

            const transcriptionItem = document.createElement('p');
            transcriptionItem.innerText = `Iteration ${message.iteration}: ${message.text}`;
            transcriptionList.appendChild(transcriptionItem);
            transcriptionList.scrollTop = transcriptionList.scrollHeight;
        } else if (message.event === 'special_message') {
            showSpecialMessage(message.text, message.link);
        } else if (message.event === 'complete') {
            progressDiv.innerText = 'Complete';
            finalTextP.innerText = message.text;
            
            // Prompt the user to record the sentence again
            const recordAgainBtn = document.createElement('button');
            recordAgainBtn.classList.add('btn');
            recordAgainBtn.innerText = 'Record the Sentence to Continue';
            recordAgainBtn.addEventListener('click', () => {
                // Reset UI elements
                transcriptionList.innerHTML = '';
                finalTextP.innerText = '';
                recordAgainBtn.remove();
                // Start recording
                startAudioBtn.click();
            });
            document.getElementById('result').appendChild(recordAgainBtn);
            websocket.close();
        } else if (message.event === 'error') {
            progressDiv.innerText = `Error: ${message.message}`;
            websocket.close();
        }
    };

    websocket.onerror = (event) => {
        progressDiv.innerText = 'An error occurred. Please try again later.';
        console.error('WebSocket error:', event);
    };

    websocket.onclose = (event) => {
        if (event.code !== 1000) { // 1000 indicates a normal closure
            progressDiv.innerText = 'Connection closed unexpectedly. Please refresh the page.';
        }
        console.log('WebSocket closed');
    };
}

startAudioBtn.addEventListener('click', async () => {
    startAudioBtn.disabled = true;
    stopAudioBtn.disabled = false;
    audioChunks = [];

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const numCharacters = parseInt(numCharactersInput.value);
            startProcess({ num_characters: numCharacters });
        };

        mediaRecorder.start();
    } catch (error) {
        alert('Could not access microphone. Please check your settings.');
        startAudioBtn.disabled = false;
        stopAudioBtn.disabled = true;
    }
});

stopAudioBtn.addEventListener('click', () => {
    mediaRecorder.stop();
    startAudioBtn.disabled = false;
    stopAudioBtn.disabled = true;
});

function showSpecialMessage(text, link) {
    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'specialMessageOverlay';
    overlay.innerHTML = `
        <div class="special-message-content">
            <p>${text}</p>
            <a href="${link}" target="_blank">Visit Alon's LinkedIn</a>
        </div>
    `;
    document.body.appendChild(overlay);

    // Add zoom-in animation class
    overlay.classList.add('zoom-in');

    // Remove the overlay after a delay
    setTimeout(() => {
        overlay.classList.add('fade-out');
        overlay.addEventListener('animationend', () => {
            if (overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        });
    }, 7000); // Display duration in milliseconds
}
function playAudio(audioUrl, volumeLevel) {
    const audio = new Audio(audioUrl);
    if (volumeLevel === 'quiet') {
        audio.volume = 0.8; // Adjust volume as needed
    } else {
        audio.volume = 1.0;
    }
    audio.play();
}