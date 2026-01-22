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

        console.log("Received from Python: ", data);
        alert("Success. Look the JSON in the console (F12)");
    } catch (error){
        console.log("Error: ", error);
        alert("Error to Send File");
    }
}
