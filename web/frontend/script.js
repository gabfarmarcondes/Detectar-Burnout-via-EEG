const API_URL = "http://127.0.0.1:8000/predict";

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultCard = document.getElementById('result-card');
const resultTitle = document.getElementById('result-title');
const distBarFill = document.getElementById('dist-bar');
const visualizer = document.querySelector('.distance-visualizer');


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
    
    // 1. Prepara a tela
    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({behavior: 'smooth'});
    visualizer.classList.add('loading');
    
    resultTitle.style.color = "#94a3b8";
    
    // 2. Configura a Barra de Progresso
    let progress = 0;
    distBarFill.style.width = "0%";
    distBarFill.style.transition = "width 0.2s linear";
    
    const progressInterval = setInterval(() => {
        let increment;

        if (progress < 20) {
            increment = Math.random() * 0.8; 
        } 
        else if (progress < 50) {
            increment = Math.random() * 0.3; 
        } 
        else {
            increment = Math.random() * 0.1; 
        }

        progress += increment;
        
        if (progress > 90) progress = 90;
        
        distBarFill.style.width = `${progress}%`;
        resultTitle.innerText = `Processing... ${Math.round(progress)}%`;

    }, 100);

    // 4. Envio Real para a API
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Aguarda o Python
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("API ERROR");

        const data = await response.json();
        
        // 5. Finalização (O Grand Finale)
        clearInterval(progressInterval); // Para o loop falso
        
        // Força ir para 100% visualmente
        distBarFill.style.transition = "width 1.2s cubic-bezier(0.22, 1, 0.36, 1)";
        distBarFill.style.width = "100%";
        resultTitle.innerText = "Finalizing Analysis... 100%";
        setTimeout(() => {
            visualizer.classList.remove('loading');
            distBarFill.style.background = "";

            displayResults(data);
        }, 1200); 

    } catch (error){
        clearInterval(progressInterval);
        console.log("Error: ", error);
        resultTitle.innerText = "Error: " + error.message;
        resultTitle.style.color = "#ef4444";
        distBarFill.style.width = "0%";
    }
}

function displayResults(data){
    // Elementos do HTML
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const confValue = document.getElementById('confidence-value');
    const winValue = document.getElementById('window-value');
    const distMarker = document.getElementById('dist-marker');

    const prediction = data.prediction;
    const confidence = data.confidence;
    const windows = data.details.windows_analyzed;

    resultCard.classList.remove('hidden');

    resultCard.scrollIntoView({behavior: 'smooth'});

    const isBurnout = prediction === "Burnout";
    const color = isBurnout ? "#ef4444" : "#10b981";
    const text = isBurnout ? "Burnout Detected" : "Relaxed State"

    resultTitle.innerText = text;
    resultTitle.style.color = color;

    confValue.innerText = confidence;
    winValue.innerText = windows;

    let rawConfidence = confidence.replace(/[^0-9.]/g, ''); 
    let percent = parseFloat(rawConfidence);

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

    const plotContainer = document.getElementById('spatial-plot-container');
    const plotImg = document.getElementById('spatial-plot-img');
    const imageSource = data.image_base64 || (data.details && data.details.image_base64);

    if (imageSource) {
        plotImg.src = "data:image/png;base64," + imageSource;
        plotContainer.classList.remove('hidden');
    } else {
        console.warn("image_base64 not found in JSON:", data);
    }

    const xaiContainer = document.getElementById('xai-container');
    const xaiImg = document.getElementById('xai-img');

    // Tenta pegar da raiz ou details
    const xaiSource = data.xai_base64 || (data.details && data.details.xai_base64);

    if (xaiSource) {
        xaiImg.src = "data:image/png;base64," + xaiSource;
        xaiContainer.classList.remove('hidden');
    }
}