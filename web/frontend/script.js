const API_URL = "http://127.0.0.1:8000/predict";

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');


// Previne que o computador abra o arquivo
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefault, false);
})

function preventDefault(e){
    e.preventDefault();
    e.stopPropagation();
}

// Efeitos visuais quando arrasta por cima
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

// Soltar o arquivo
dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e){
    e.preventDefault();
    e.stopPropagation();

    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files){
    if (files.length > 0) {
        const file = files[0];
        if (file.name.endsWith('.txt')) {
            uploadFile(file);
        } else {
            alert("Please upload a valid .txt file from OpenBCI/STEW.");
        }
    }
}

async function uploadFile(file) {
    // Prepara o envio
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Envia para API Python
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("API ERROR");

        const data = await response.json();
        displayResults(data);

        console.log("Received from Python: ", data);
    } catch (error){
        console.log("Error: ", error);
        alert("Error to Process: " + error.message);
    }
}

function displayResults(data){
    // Elementos do HTML
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const statusPulse = document.getElementById('status-pulse');
    const confValue = document.getElementById('confidence-value');
    const winValue = document.getElementById('window-value');
    const distMarker = document.getElementById('dist-marker');

    // Extrair os dados do JSON vistos no console
    const prediction = data.prediction;
    const confidence = data.confidence;
    const windows = data.details.windows_analyzed;

    // Mostrar o Card
    resultCard.classList.remove('hidden');

    // Rolar a tela até o resultado
    resultCard.scrollIntoView({behavior: 'smooth'});

    // Lógica de cores
    const isBurnout = prediction === "Burnout";
    const color = isBurnout ? "#ef4444" : "#10b981";
    const text = isBurnout ? "Burnout Detected" : "Relaxed State"

    // Atualizar o DOM
    resultTitle.innerText = text;
    resultTitle.style.color = color;

    confValue.innerText = confidence;
    winValue.innerText = windows;

    let rawConfidence = confidence.replace(/[^0-9.]/g, ''); 
    let percent = parseFloat(rawConfidence);

    // Se for Burnout (lado direito da barra), soma. Se Relaxado (esquerdo), subtrai.
    let position = 50; 
    if (prediction === "Burnout") {
        // Ex: 50 + (90 / 2) = 95% (Direita)
        position = 50 + (percent / 2.2); // Dividi por 2.2 pra não colar na borda
    } else {
        // Ex: 50 - (90 / 2) = 5% (Esquerda)
        position = 50 - (percent / 2.2);
    }

    // Trava limites (Segurança)
    position = Math.max(2, Math.min(position, 98));

    // Aplica
    distMarker.style.left = `${position}%`;
}