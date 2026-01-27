const API_URL = "/predict";

document.addEventListener('DOMContentLoaded', () => {
    console.log("App Loaded");

    const dropZone = document.getElementById('drop-zone');

    if (dropZone) {
        const fileInput = document.getElementById('file-input');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefault, false);
        });

        function preventDefault(e){
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
        });

        dropZone.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', function() { handleFiles(this.files); });

        function handleDrop(e){
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }

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
    }

const cards = document.querySelectorAll('.card');

    if (cards.length > 0) {
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        let currentIndex = 0;

        // Função para atualizar as classes com lógica Coverflow
        function updateCarousel() {
            cards.forEach((card, index) => {
                // Remove todas as classes de estado
                card.classList.remove('active', 'prev', 'next', 'hidden-left', 'hidden-right');
                
                // Calcula a distância do card atual para o índice central
                let distance = index - currentIndex;
                
                // Ajusta para lógica circular (ex: se tiver 6 cards, o último é -1 do primeiro)
                if (distance > cards.length / 2) distance -= cards.length;
                if (distance < -cards.length / 2) distance += cards.length;

                // Aplica as classes baseadas na distância
                if (distance === 0) {
                    card.classList.add('active');
                } else if (distance === 1) {
                    card.classList.add('next');
                } else if (distance === -1) {
                    card.classList.add('prev');
                } else if (distance > 1) {
                    card.classList.add('hidden-right');
                } else if (distance < -1) {
                    card.classList.add('hidden-left');
                }
            });
        }

        // Funções de Navegação
        const goNext = () => {
            currentIndex = (currentIndex + 1) % cards.length;
            updateCarousel();
        };

        const goPrev = () => {
            currentIndex = (currentIndex - 1 + cards.length) % cards.length;
            updateCarousel();
        };

        // Event Listeners dos Botões
        if (nextBtn) nextBtn.addEventListener('click', goNext);
        if (prevBtn) prevBtn.addEventListener('click', goPrev);

        // Clique nos próprios cards (Prev/Next) para navegar
        cards.forEach((card, index) => {
            card.addEventListener('click', () => {
                let distance = index - currentIndex;
                if (distance > cards.length / 2) distance -= cards.length;
                if (distance < -cards.length / 2) distance += cards.length;

                if (distance === 1) goNext();
                if (distance === -1) goPrev();
            });
        });

        // Inicializa
        updateCarousel();
    }
});

async function uploadFile(file) {
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const distBarFill = document.getElementById('dist-bar');
    const visualizer = document.querySelector('.distance-visualizer');

    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({behavior: 'smooth'});
    visualizer.classList.add('loading');
    resultTitle.style.color = "#94a3b8";

    let progress = 0;
    distBarFill.style.width = "0%";
    distBarFill.style.transition = "width 0.2s linear";
    
    const progressInterval = setInterval(() => {
        let increment;
        if (progress < 20) increment = Math.random() * 0.8; 
        else if (progress < 50) increment = Math.random() * 0.3; 
        else increment = Math.random() * 0.1; 

        progress += increment;
        if (progress > 90) progress = 90;
        
        distBarFill.style.width = `${progress}%`;
        resultTitle.innerText = `Processing... ${Math.round(progress)}%`;
    }, 100);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("API ERROR");

        const data = await response.json();

        clearInterval(progressInterval);
        
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
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const confValue = document.getElementById('confidence-value');
    const winValue = document.getElementById('window-value');
    const distMarker = document.getElementById('dist-marker');

    const prediction = data.prediction;
    const confidence = data.confidence;
    const windows = data.details.windows_analyzed;

    resultCard.classList.remove('hidden');
    const isBurnout = prediction === "Burnout";
    resultTitle.innerText = isBurnout ? "Burnout Detected" : "Relaxed State";
    resultTitle.style.color = isBurnout ? "#ef4444" : "#10b981";

    confValue.innerText = confidence;
    winValue.innerText = windows;

    let percent = parseFloat(confidence.replace(/[^0-9.]/g, ''));
    let position = prediction === "Burnout" ? 50 + (percent / 2.2) : 50 - (percent / 2.2);
    position = Math.max(2, Math.min(position, 98));
    distMarker.style.left = `${position}%`;

    const setImg = (id, containerId, source) => {
        if (source) {
            document.getElementById(id).src = "data:image/png;base64," + source;
            document.getElementById(containerId).classList.remove('hidden');
        }
    };

    setImg('spatial-plot-img', 'spatial-plot-container', data.image_base64 || data.details?.image_base64);
    setImg('xai-img', 'xai-container', data.xai_base64 || data.details?.xai_base64);
    setImg('topomap-img', 'topomap-container', data.topomap_base64 || data.details?.topomap_base64);
}